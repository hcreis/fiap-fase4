import matplotlib.pyplot as plt
from deepface import DeepFace

import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict, deque

from insightface.app import FaceAnalysis
from ultralytics import YOLO
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

# ======================================================
# CONFIGURAÇÕES GERAIS
# ======================================================
PULAR_FRAMES = 0  # 0 = não pula
FONTE_PADRAO = cv2.FONT_HERSHEY_DUPLEX
FONTE = FONTE_PADRAO
ESCALA_FONTE = 0.75
ESPESSURA_FONTE = 2
COR_TEXTO = (255, 255, 255)
COR_CAIXA = (0, 255, 0)

# Emotions consideradas negativas para anomalia emocional
EMOCOES_NEGATIVAS = {"sad", "angry", "fear", "disgust"}

# Identidade facial temporal (mais rígido -> menos "parecidos" viram a mesma pessoa)
LIMIAR_SIMILARIDADE_FACE = 0.35  # era 0.55; aumente para ficar mais rígido (0.60~0.65)
MIN_FRAMES_PARA_CONFIRMAR = 360  # exige consistência antes de "fixar" embedding
MAX_FRAMES_SEM_MATCH = 20

# Cenas (lógica enxuta)
# Só troca de cena se o novo dominante permanecer por X frames consecutivos
MIN_FRAMES_TROCA_CENA = 120  # ~1 segundo a 30fps; ajuste conforme vídeo
# Pequena histerese: mantém cena atual se dominante sumir por poucos frames
TOLERANCIA_SEM_DOMINANTE = 10  # frames sem dominante antes de encerrar/avaliar troca

# Pose / Aceno
MAX_HISTORICO_ACENO = 15

# Aperto de mão: janela de confirmação (para reduzir falsos positivos)
MIN_FRAMES_APERTO_MAO = 1  # precisa ocorrer por ~0.4s para contar como evento
DISTANCIA_APERTO_MAO_PX = 16

# Anomalia de movimento
# Detecta "picos" de velocidade de punhos/torso muito acima do padrão por personagem
JANELA_VELOCIDADE = 10
Z_SCORE_ANOMALIA = 3.0

# ======================================================
# LANDMARKS (APENAS DESENHO - MEDIAPIPE COM SKELETON)
# ======================================================
DESENHAR_LANDMARKS_FACIAIS = True
DESENHAR_LANDMARKS_CORPORAIS = True

mp_face = mp.solutions.face_mesh
face_mesh_draw = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

mp_pose_draw = mp.solutions.pose
pose_draw = mp_pose_draw.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ======================================================
# MODELOS
# ======================================================
modelo_pose = YOLO("yolo11n-pose.pt")


# ======================================================
# EMOÇÕES (ONNX – substitui modelo ONNX de emoções)
# ======================================================
EMOCOES = [
    "neutral",
    "happy",
    "surprise",
    "sad",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

def classificar_emocao_deepface(face_bgr):
    try:
        res = DeepFace.analyze(
            face_bgr,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="skip"
        )
        return res[0]["dominant_emotion"]
    except Exception:
        return "neutral"


# ======================================================
# IDENTIDADES TEMPORAIS (sem imagens de referência)
# ======================================================
identidades_conhecidas = []  # [{nome, emb, frames, frames_sem_match}]
contador_pessoas = 1


def resolver_identidade(embedding):
    global contador_pessoas

    if embedding is None or embedding.size == 0:
        return "DESCONHECIDO"

    if not identidades_conhecidas:
        nome = f"PERSON_{contador_pessoas}"
        contador_pessoas += 1
        identidades_conhecidas.append(
            {
                "nome": nome,
                "emb": embedding,
                "frames": 1,
            }
        )
        return nome

    sims = [float(np.dot(i["emb"], embedding)) for i in identidades_conhecidas]
    idx = int(np.argmax(sims))
    melhor_sim = sims[idx]
    ident = identidades_conhecidas[idx]

    if melhor_sim >= LIMIAR_SIMILARIDADE_FACE:
        ident["frames"] += 1

        # EMA moderado
        ident["emb"] = (0.9 * ident["emb"] + 0.1 * embedding).astype(np.float32)
        ident["emb"] /= np.linalg.norm(ident["emb"]) + 1e-9

        return ident["nome"]

    # NÃO há match → nova identidade
    nome = f"PERSON_{contador_pessoas}"
    contador_pessoas += 1
    identidades_conhecidas.append(
        {
            "nome": nome,
            "emb": embedding,
            "frames": 1,
        }
    )
    return nome


# ======================================================
# INSIGHTFACE
# ======================================================
def iniciar_insightface():
    # Mantendo CPU para evitar problemas com CUDA/onnxruntime no seu ambiente atual.
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


# ======================================================
# ATIVIDADES (POSE)
# ======================================================
historico_aceno = {}  # track_id -> list[x]


def classificar_em_pe_ou_sentado(kps):
    if kps is None or kps.shape[0] < 16:
        return None

    quadril = kps[11]   # hip
    joelho = kps[13]   # knee
    tornozelo = kps[15]  # ankle

    # confiança mais rígida
    if min(quadril[2], joelho[2], tornozelo[2]) < 0.5:
        return None

    dy_hk = quadril[1] - joelho[1]
    dy_ka = joelho[1] - tornozelo[1]
    dy_ha = quadril[1] - tornozelo[1]

    # sentado: quadril e joelho muito próximos
    if abs(dy_hk) < abs(dy_ka) * 0.5:
        return "SENTADO"

    # EM_PE rígido
    if (
        dy_hk < -10 and               # quadril acima do joelho
        dy_ka < -10 and               # joelho acima do tornozelo
        abs(dy_ha) > 0.25 * abs(dy_ka) and  # perna estendida
        abs(quadril[0] - tornozelo[0]) < 0.25 * abs(dy_ha)  # postura vertical
    ):
        return "EM_PE"

    return None


def detectar_aceno(track_id, kps):
    # precisa ter pelo menos 11 keypoints (punho = 10)
    if kps is None or kps.shape[0] <= 10:
        return False

    punho = kps[10]
    ombro = kps[6]

    # valida confiança
    if punho[2] < 0.4 or ombro[2] < 0.4:
        return False

    # punho precisa estar acima do ombro
    if punho[1] > ombro[1]:
        return False

    hist = historico_aceno.setdefault(track_id, [])
    hist.append(float(punho[0]))

    if len(hist) > MAX_HISTORICO_ACENO:
        hist.pop(0)

    if len(hist) > 10:
        dx = np.diff(hist)
        # alternância de direção (movimento lateral)
        return int(np.sum(np.diff(np.sign(dx)) != 0)) >= 2

    return False


# ======================================================
# APERTO DE MÃO (MÃOS PRÓXIMAS – MEDIAPIPE HANDS)
# ======================================================
DIST_MAO_PX = 50

def centro_mao(hand_landmarks, w, h):
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    return (int(np.mean(xs) * w), int(np.mean(ys) * h))


def detectar_aperto_de_mao(centros_maos):
    pontos = []

    for c in centros_maos:
        if isinstance(c, dict):
            if "x" in c and "y" in c:
                pontos.append((c["x"], c["y"]))
        elif isinstance(c, (tuple, list)) and len(c) == 2:
            pontos.append((c[0], c[1]))

    for i in range(len(pontos)):
        for j in range(i + 1, len(pontos)):
            if (
                10 < np.linalg.norm(np.array(pontos[i]) - np.array(pontos[j]))
                < DISTANCIA_APERTO_MAO_PX
            ):
                return True

    return False


# ======================================================
# ANOMALIA DE MOVIMENTO (simples)
# ======================================================
# Mantém histórico de velocidades por personagem dominante (ou por track se preferir)
historico_movimento = defaultdict(lambda: deque(maxlen=JANELA_VELOCIDADE))
# Para estimar velocidade, guardamos último ponto (punho direito e quadril) por "pessoa"
ultimo_ponto = {}  # chave -> (x, y)


def registrar_movimento(chave, ponto_xy):
    """
    Atualiza velocidade instantânea com base na diferença entre frames.
    Armazena em historico_movimento[chave] para detectar outliers.
    """
    if ponto_xy is None:
        return None

    if chave in ultimo_ponto:
        dx = ponto_xy[0] - ultimo_ponto[chave][0]
        dy = ponto_xy[1] - ultimo_ponto[chave][1]
        vel = float(np.sqrt(dx * dx + dy * dy))
    else:
        vel = 0.0

    ultimo_ponto[chave] = ponto_xy
    historico_movimento[chave].append(vel)
    return vel


def checar_anomalia_movimento(chave):
    """
    Marca anomalia se a velocidade atual for muito acima do padrão da janela recente.
    """
    vals = list(historico_movimento[chave])
    if len(vals) < JANELA_VELOCIDADE:
        return False
    media = float(np.mean(vals))
    desvio = float(np.std(vals)) + 1e-9
    atual = vals[-1]
    z = (atual - media) / desvio
    return z >= Z_SCORE_ANOMALIA


# ======================================================
# CENAS (POR PERSONAGEM DOMINANTE COM ESTABILIDADE)
# ======================================================
cenas = []
cena_atual = None

# estado para troca com consistência
candidato_dominante = None
frames_candidato = 0
frames_sem_dominante = 0


def obter_personagem_dominante(lista_faces):
    """
    Personagem dominante = maior face (maior área de bbox) no frame.
    """
    if not lista_faces:
        return None
    return max(
        lista_faces,
        key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
    )["nome"]


def iniciar_cena(personagem):
    return {
        "personagem": personagem,
        "frames": 0,
        "atividades": Counter(),
        "emocoes": Counter(),
        "anomalias_emocao": 0,
        "anomalias_movimento": 0,
        "eventos": Counter(),  # eventos “compactados”: APERTO_DE_MAO, ACENO (confirmados por janela)
    }


# Para “compactar” eventos e evitar supercontagem frame-a-frame
contador_aperto_mao = 0
contador_aceno = 0

# ======================================================
# ESTATÍSTICAS GLOBAIS
# ======================================================
estatisticas = {
    "frames": 0,
    "atividades": Counter(),
    "emocoes": Counter(),
    "anomalias_emocao": 0,
    "anomalias_movimento": 0,
    "eventos": Counter(),
}

sequencia_emocao = defaultdict(int)


# ======================================================
# PROCESSAMENTO DO VÍDEO
# ======================================================
def processar_video(caminho_video, caminho_saida_video):

    global cena_atual, candidato_dominante, frames_candidato, frames_sem_dominante
    global contador_aperto_mao, contador_aceno

    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {caminho_video}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        caminho_saida_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (largura, altura),
    )

    app_face = iniciar_insightface()
    id_pose = 0

    for frame_idx in tqdm(range(total), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        # if(frame_idx < 97 or frame_idx > 102):continue

        faces_data = []
        faces = app_face.get(frame) or []
        numero_pessoas = len(faces)

        # ======================================================
        # LANDMARKS CORPORAIS (SKELETON - MEDIAPIPE)
        # ======================================================
        if DESENHAR_LANDMARKS_CORPORAIS and numero_pessoas == 1:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res_pose_draw = pose_draw.process(frame_rgb)

            if res_pose_draw.pose_landmarks:
                # Skeleton
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=res_pose_draw.pose_landmarks,
                    connections=mp_pose_draw.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 0, 255), thickness=2
                    ),
                )

                # Pontos (opcional – se quiser manter)
                for lm in res_pose_draw.pose_landmarks.landmark:
                    x_lm = int(lm.x * largura)
                    y_lm = int(lm.y * altura)
                    cv2.circle(frame, (x_lm, y_lm), 2, (0, 255, 0), -1)

        # pular frames se desejado
        if PULAR_FRAMES and (estatisticas["frames"] % (PULAR_FRAMES + 1) != 0):
            estatisticas["frames"] += 1
            out.write(frame)
            continue

        estatisticas["frames"] += 1

        # ======================================================
        # FACE + EMOÇÃO
        # ======================================================

        for face in faces:
            emb = face.normed_embedding.astype(np.float32)
            nome = resolver_identidade(emb)

            x1, y1, x2, y2 = map(int, face.bbox)
            faces_data.append({"nome": nome, "bbox": (x1, y1, x2, y2)})

            # desenho bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), COR_CAIXA, 2)

            # emoção (modelo ONNX de emoções) em ROI
            roi = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]

            # ======================================================
            # LANDMARKS FACIAIS (MEDIAPIPE – DESENHO APENAS)
            # ======================================================
            if DESENHAR_LANDMARKS_FACIAIS and roi.size > 0:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                res_face_draw = face_mesh_draw.process(roi_rgb)
                if res_face_draw.multi_face_landmarks:
                    h_roi, w_roi = roi.shape[:2]

                    for face_landmarks in res_face_draw.multi_face_landmarks:
                        # Contorno do rosto
                        mp_drawing.draw_landmarks(
                            image=roi,
                            landmark_list=face_landmarks,
                            connections=mp_face.FACEMESH_FACE_OVAL,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(
                                color=(0, 255, 0), thickness=1, circle_radius=1
                            ),
                        )

                        # Pontos individuais (opcional)
                        for lm in face_landmarks.landmark:
                            px = int(lm.x * w_roi)
                            py = int(lm.y * h_roi)
                            cv2.circle(roi, (px, py), 1, (255, 255, 0), -1)

            if roi.size == 0:
                continue

            try:
                emo = classificar_emocao_deepface(roi)

                estatisticas["emocoes"][emo] += 1
                if cena_atual:
                    cena_atual["emocoes"][emo] += 1

                # anomalia emocional (streak)
                if emo in EMOCOES_NEGATIVAS:
                    sequencia_emocao[nome] += 1
                    if sequencia_emocao[nome] > 15:
                        estatisticas["anomalias_emocao"] += 1
                        if cena_atual:
                            cena_atual["anomalias_emocao"] += 1
                        sequencia_emocao[nome] = 0
                else:
                    sequencia_emocao[nome] = 0

                cv2.putText(
                    frame,
                    f"{nome} - {emo}",
                    (x1, y2 + 20),
                    FONTE,
                    0.7,
                    COR_TEXTO,
                    2,
                )

            except Exception:
                pass

        # ======================================================
        # POSE / ATIVIDADES
        # ======================================================
        resultado_pose = modelo_pose.predict(frame, verbose=False)[0]
        pessoas_pose = []

        if (
            resultado_pose.keypoints is not None
            and resultado_pose.keypoints.xy is not None
        ):
            kxy = resultado_pose.keypoints.xy.cpu().numpy()

            # conf pode ser None em alguns frames
            if resultado_pose.keypoints.conf is not None:
                kcf = resultado_pose.keypoints.conf.cpu().numpy()
            else:
                # cria conf = 1.0 para todos os keypoints detectados
                kcf = np.ones((kxy.shape[0], kxy.shape[1]), dtype=np.float32)

            for i in range(len(kxy)):
                kps = np.concatenate([kxy[i], kcf[i][..., None]], axis=1)
                pessoas_pose.append({"id": id_pose, "kps": kps})
                id_pose += 1

        # Ações frame-a-frame
        houve_aceno = False
        atividades_frame = set()

        for p in pessoas_pose:
            pid = p["id"]
            kps = p["kps"]

            act = classificar_em_pe_ou_sentado(kps)
            if act:
                atividades_frame.add(act)
                estatisticas["atividades"][act] += 1
                if cena_atual:
                    cena_atual["atividades"][act] += 1

            if detectar_aceno(pid, kps):
                houve_aceno = True

            # anomalia de movimento: usamos punho direito se disponível; senão quadril
            ponto = None
            # valida tamanho mínimo (punho = 10, quadril = 11)
            if kps is not None and kps.shape[0] > 11:
                punho = kps[10]
                quadril = kps[11]

                if punho[2] > 0.4:
                    ponto = (float(punho[0]), float(punho[1]))
                elif quadril[2] > 0.4:
                    ponto = (float(quadril[0]), float(quadril[1]))

            # chave = personagem dominante atual (para ficar “acadêmico” e explicar no texto)
            # se não houver cena atual, usa "GLOBAL"
            chave_mov = cena_atual["personagem"] if cena_atual else "GLOBAL"
            registrar_movimento(chave_mov, ponto)
            if checar_anomalia_movimento(chave_mov):
                estatisticas["anomalias_movimento"] += 1
                if cena_atual:
                    cena_atual["anomalias_movimento"] += 1

        # ======================================================
        # APERTO DE MÃO (MÃOS PRÓXIMAS – MEDIAPIPE HANDS)
        # ======================================================
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res_hands = hands.process(frame_rgb)

        centros_maos = []
        if res_hands.multi_hand_landmarks:
            for hand in res_hands.multi_hand_landmarks:
                cx, cy = centro_mao(hand, largura, altura)
                centros_maos.append((int(cx), int(cy)))

        houve_aperto_mao = len(centros_maos) >= 2 and detectar_aperto_de_mao(
            centros_maos
        )

        # ======================================================
        # EVENTOS “CONFIRMADOS” (reduz falsos positivos)
        # ======================================================
        # ACENO
        if houve_aceno:
            contador_aceno += 1
        else:
            if contador_aceno >= MIN_FRAMES_PARA_CONFIRMAR:
                estatisticas["eventos"]["ACENO"] += 1
                if cena_atual:
                    cena_atual["eventos"]["ACENO"] += 1
            contador_aceno = 0

        # APERTO DE MÃO
        if houve_aperto_mao:
            contador_aperto_mao += 1
        else:
            if contador_aperto_mao >= MIN_FRAMES_APERTO_MAO:
                estatisticas["eventos"]["APERTO_DE_MAO"] += 1
                if cena_atual:
                    cena_atual["eventos"]["APERTO_DE_MAO"] += 1
            contador_aperto_mao = 0

        # Desenho de “atividades” no frame (texto)
        y = 30
        for txt in sorted(list(atividades_frame)):
            cv2.putText(frame, txt, (20, y), FONTE, 0.7, COR_CAIXA, 2)
            y += 25

        # ======================================================
        # CENAS: por personagem dominante + estabilidade temporal
        # ======================================================
        dominante = obter_personagem_dominante(faces_data)

        if dominante is None:
            frames_sem_dominante += 1
        else:
            frames_sem_dominante = 0

        # se não há cena atual e apareceu um dominante, inicia
        if cena_atual is None and dominante is not None:
            cena_atual = iniciar_cena(dominante)
            candidato_dominante = None
            frames_candidato = 0

        # se há cena atual e existe dominante
        if cena_atual is not None and dominante is not None:
            if dominante == cena_atual["personagem"]:
                # dominante consistente com a cena atual
                candidato_dominante = None
                frames_candidato = 0
            else:
                # dominante diferente -> começa/continua candidato
                if candidato_dominante != dominante:
                    candidato_dominante = dominante
                    frames_candidato = 1
                else:
                    frames_candidato += 1

                # só troca cena se candidato persistir por tempo mínimo
                if frames_candidato >= MIN_FRAMES_TROCA_CENA:
                    cenas.append(cena_atual)
                    cena_atual = iniciar_cena(candidato_dominante)
                    candidato_dominante = None
                    frames_candidato = 0

        # se ficamos muito tempo sem dominante, não troca cena automaticamente; apenas mantém.
        # opcionalmente você pode encerrar cena se quiser (não recomendo para relatório enxuto)
        if cena_atual is not None:
            cena_atual["frames"] += 1

        out.write(frame)

    # fecha eventos pendentes no final
    if contador_aceno >= MIN_FRAMES_PARA_CONFIRMAR:
        estatisticas["eventos"]["ACENO"] += 1
        if cena_atual:
            cena_atual["eventos"]["ACENO"] += 1

    if contador_aperto_mao >= MIN_FRAMES_APERTO_MAO:
        estatisticas["eventos"]["APERTO_DE_MAO"] += 1
        if cena_atual:
            cena_atual["eventos"]["APERTO_DE_MAO"] += 1

    if cena_atual is not None:
        cenas.append(cena_atual)

    cap.release()
    out.release()


# ======================================================
# RELATÓRIOS
# ======================================================
def escrever_relatorio_tecnico(caminho, fps=30):
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO TÉCNICO – ANÁLISE GLOBAL DO VÍDEO\n\n")

        # ======================================================
        # DURAÇÃO TOTAL
        # ======================================================
        total_frames = estatisticas["frames"]
        duracao_s = total_frames / fps if fps else 0

        f.write(f"Duração total analisada: {duracao_s:.1f}s\n")
        f.write(f"Total de frames processados: {total_frames}\n\n")

        # ======================================================
        # ATIVIDADES (PERCENTUAL POR FRAME)
        # ======================================================
        f.write("Atividade corporal (percentual do tempo):\n")
        if estatisticas["atividades"]:
            total_atividades = sum(estatisticas["atividades"].values())

            for k, v in estatisticas["atividades"].most_common():
                perc = (v / total_atividades) * 100
                f.write(f"  - {k}: {perc:.1f}%\n")
        f.write("\n")

        # ======================================================
        # EMOÇÕES (PERCENTUAL POR FACE ANALISADA)
        # ======================================================
        f.write("Estado emocional (percentual de ocorrência):\n")
        if estatisticas["emocoes"]:
            total_emocoes = sum(estatisticas["emocoes"].values())

            for k, v in estatisticas["emocoes"].most_common():
                perc = (v / total_emocoes) * 100
                f.write(f"  - {k.capitalize()}: {perc:.1f}%\n")
        else:
            f.write("  - Não determinado\n")

        f.write("\n")

        # ======================================================
        # EVENTOS
        # ======================================================
        f.write("Eventos relevantes detectados:\n")
        if estatisticas["eventos"]:
            for k, v in estatisticas["eventos"].most_common():
                f.write(f"  - {k}: {v}\n")
        else:
            f.write("  - Nenhum\n")

        f.write("\n")

        # ======================================================
        # ANOMALIAS
        # ======================================================
        f.write("Anomalias detectadas:\n")
        f.write(f"  - Emoção: {estatisticas['anomalias_emocao']}\n")
        f.write(f"  - Movimento: {estatisticas['anomalias_movimento']}\n")

def gerar_grafico_emocoes(caminho_saida):
    if not estatisticas["emocoes"]:
        return

    labels = []
    sizes = []

    total = sum(estatisticas["emocoes"].values())

    for emo, count in estatisticas["emocoes"].most_common():
        labels.append(emo.capitalize())
        sizes.append((count / total) * 100)

    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Distribuição Global de Emoções")
    plt.axis("equal")

    plt.tight_layout()
    plt.savefig(caminho_saida)
    plt.close()

# ======================================================
# MAIN
# ======================================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    caminho_video_entrada = os.path.join(base_dir, "video_tech_challenger_fase_4.mp4")
    caminho_video_saida = os.path.join(
        base_dir, "video_tech_challenger_fase_4_final.mp4"
    )

    # Verifica se vídeo de entrada existe
    if not os.path.exists(caminho_video_entrada):
        print(f"Arquivo de entrada não encontrado: {caminho_video_entrada}")
        print("Coloque o vídeo na raiz do projeto ou edite o caminho no script.")
        return

    # Captura FPS real para relatório
    cap_tmp = cv2.VideoCapture(caminho_video_entrada)
    fps = int(cap_tmp.get(cv2.CAP_PROP_FPS)) or 30
    cap_tmp.release()

    processar_video(caminho_video_entrada, caminho_video_saida)

    relatorio_tecnico = os.path.join(base_dir, "relatorio_final_tecnico.txt")

    # Passa o FPS real para o relatório técnico
    escrever_relatorio_tecnico(relatorio_tecnico, fps=fps)

    grafico_path = os.path.join(base_dir, "grafico_emocoes.png")
    gerar_grafico_emocoes(grafico_path)

    print("\nArquivos gerados:")
    print(f"- Vídeo anotado: {caminho_video_saida}")
    print(f"- Relatório técnico: {relatorio_tecnico}")
    print(f"- Gráfico de emoções: {grafico_path}")

if __name__ == "__main__":
    main()
