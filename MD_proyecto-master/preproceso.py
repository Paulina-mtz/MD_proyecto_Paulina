import os
import re
import ast
import html
import argparse
import unicodedata
import pandas as pd
import numpy as np

# ====== Dependencias opcionales ======
try:
    import emoji
    HAVE_EMOJI = True
except Exception:
    HAVE_EMOJI = False

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    HAVE_LANGDETECT = True
except Exception:
    HAVE_LANGDETECT = False

# ====== Embeddings opcionales ======
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SBERT = True
except Exception:
    HAVE_SBERT = False


# ====== Expresiones regulares y utilidades ======
URL_RE = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
USER_RE = re.compile(r'(^|[^A-Za-z0-9_])@[A-Za-z0-9_]+')
SUB_RE  = re.compile(r'(^|[^A-Za-z0-9_])r/[A-Za-z0-9_]+')
HASHTAG = re.compile(r'#([A-Za-z0-9_]+)')
MULTISPACE = re.compile(r'\s+')
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
ABBREV_FIX = re.compile(r'\b(e\.g\.|i\.e\.|mr\.|mrs\.|dr\.|vs\.|etc\.)', re.IGNORECASE)

def normalize_unicode(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def demojize(text: str) -> str:
    if HAVE_EMOJI:
        # 🙂 -> :slightly_smiling_face:
        return emoji.demojize(text, language='en')
    return text  # fallback: sin cambio si no hay emoji

def clean_text_for_bert(text: str) -> str:
    """
    Limpieza 'ligera' segura para modelos BERT uncased:
    - decodifica HTML
    - normaliza unicode
    - lowercase
    - placeholders para url/@/r/ y hashtags
    - emojis -> texto (si está disponible la librería)
    - colapsa espacios
    """
    if not isinstance(text, str):
        text = str(text)
    text = html.unescape(text)
    text = normalize_unicode(text)
    text = text.strip().lower()

    # proteger abreviaturas comunes (evitar splits raros)
    text = ABBREV_FIX.sub(lambda m: m.group(0).replace('.', ''), text)  # "e.g." -> "eg"

    # placeholders semánticos
    text = URL_RE.sub(" <url> ", text)
    text = USER_RE.sub(lambda m: (m.group(1) + " <user> "), text)
    text = SUB_RE.sub(lambda m: (m.group(1) + " <subreddit> "), text)
    text = HASHTAG.sub(lambda m: f" <hashtag_{m.group(1)}>", text)

    # emojis -> texto (opcional)
    text = demojize(text)

    # colapsar espacios
    text = MULTISPACE.sub(" ", text).strip()
    return text

def robust_parse_post_field(post_raw):
    """
    Interpreta 'Post' como lista si es posible; si no, devuelve una lista con 1 string.
    """
    if isinstance(post_raw, list):
        return [str(x) for x in post_raw]
    if not isinstance(post_raw, str):
        post_raw = str(post_raw)
    post_raw = post_raw.strip()
    try:
        val = ast.literal_eval(post_raw)
        if isinstance(val, (list, tuple)):
            return [str(x) for x in val]
    except Exception:
        pass
    return [post_raw]

def split_into_subsentences(text: str):
    """
    Split simple por puntuación fuerte.
    """
    text = text.replace("\n", " ").strip()
    parts = SENT_SPLIT.split(text)
    out = []
    for p in parts:
        p = p.strip(" .!?\"'`“”‘’…")
        if p:
            out.append(p)
    return out

def lang_ok(text: str, allow_set):
    if not allow_set:
        return True
    if not HAVE_LANGDETECT:
        # si no hay detector instalado, deja pasar
        return True
    try:
        lang = detect(text)
        return lang in allow_set
    except Exception:
        # en caso de duda, deja pasar o cambia a False si prefieres ser estricto
        return True

def normalize_for_dedup(text: str) -> str:
    """
    Clave de deduplicación aprox:
    - minúsculas
    - eliminar placeholders comunes
    - quitar puntuación blanda
    - colapsar espacios
    """
    t = text.lower()
    t = t.replace("<url>", "").replace("<user>", "").replace("<subreddit>", "")
    t = re.sub(r'[^\w\s]', ' ', t)
    t = MULTISPACE.sub(" ", t).strip()
    return t


# ====== Pipeline principal ======
def build_clean_dataset(input_csv: str,
                        out_csv: str,
                        min_char: int = 20,
                        lang_allow: str = "",
                        verbose: bool = True) -> pd.DataFrame:
    """
    Crea user_frase_clean.csv con columnas: user, frase, label
    - lang_allow: string con códigos separados por coma (ej. "en,es"); vacío = sin filtro
    """
    if verbose:
        print(f"[INFO] Cargando: {input_csv}")
    df = pd.read_csv(input_csv)

    allow = set([x.strip() for x in lang_allow.split(",") if x.strip()]) if lang_allow else set()

    rows = []
    for _, r in df.iterrows():
        label = r.get("Label", None)
        user_full = str(r.get("User", ""))
        user = user_full.split("-")[-1] if "-" in user_full else user_full

        posts_list = robust_parse_post_field(r.get("Post", ""))

        for frase in posts_list:
            subs = split_into_subsentences(str(frase))
            for s in subs:
                clean = clean_text_for_bert(s)
                if len(clean) < min_char:
                    continue
                if not lang_ok(clean, allow):
                    continue
                rows.append({"user": user, "frase": clean, "label": label})

    clean_df = pd.DataFrame(rows)

    # deduplicación aproximada
    if not clean_df.empty:
        clean_df["dedup_key"] = clean_df["frase"].map(normalize_for_dedup)
        before = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=["dedup_key"]).drop(columns=["dedup_key"])
        after = len(clean_df)
        if verbose:
            ratio = 0.0 if before == 0 else 1 - (after / before)
            print(f"[INFO] Deduplicación: {before} -> {after} (elim. {ratio:.2%})")

    clean_df.to_csv(out_csv, index=False)
    if verbose:
        print(f"[OK] Guardado dataset limpio: {out_csv} (filas={len(clean_df)})")
    return clean_df


def generate_embeddings(texts,
                        model_name: str,
                        device: str = None,
                        batch_size: int = 128,
                        l2_normalize: bool = False):
    if not HAVE_SBERT:
        raise RuntimeError("sentence-transformers no está instalado. Instala con: pip install sentence-transformers")

    if device is None:
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", None) else "cpu"

    print(f"[INFO] Cargando modelo: {model_name} (device={device})")
    model = SentenceTransformer(model_name, device=device)
    embs = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # L2 aparte si se solicita
    )
    if l2_normalize:
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.clip(norms, 1e-12, None)
    return embs


def main():
    parser = argparse.ArgumentParser(description="Preprocesamiento y (opcional) embeddings")
    parser.add_argument("--input", default="500_Reddit_users_posts_labels.csv",
                        help="Ruta del CSV original")
    parser.add_argument("--out-clean", default="user_frase_clean.csv",
                        help="Salida CSV limpio (user, frase, label)")
    parser.add_argument("--min-char", type=int, default=20,
                        help="Longitud mínima de subfrase")
    parser.add_argument("--lang-allow", type=str, default="",
                        help="Códigos de idioma permitidos separados por coma (ej: 'en,es'). Vacío = sin filtro")
    parser.add_argument("--do-embed", action="store_true",
                        help="Si se indica, genera embeddings con Sentence-Transformers")
    parser.add_argument("--model", default="mental/mental-bert-base-uncased",
                        help="Modelo de Sentence-Transformers a usar")
    parser.add_argument("--emb-npy", default="embeddings.npy",
                        help="Ruta de salida .npy para embeddings")
    parser.add_argument("--emb-csv", default="user_embeddings.csv",
                        help="Ruta de salida CSV con embeddings (pesado)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Tamaño de batch para encode")
    parser.add_argument("--l2", action="store_true",
                        help="Normalizar embeddings L2 tras generar")
    parser.add_argument("--no-csv-emb", action="store_true",
                        help="No guardar CSV con embeddings (solo .npy)")
    args = parser.parse_args()

    # 1) Preprocess
    clean_df = build_clean_dataset(
        input_csv=args.input,
        out_csv=args.out_clean,
        min_char=args.min_char,
        lang_allow=args.lang_allow,
        verbose=True
    )

    # 2) Embeddings opcionales
    if args.do_embed:
        if clean_df.empty:
            print("[WARN] Dataset limpio vacío. No se generarán embeddings.")
            return
        print(f"[INFO] Generando embeddings para {len(clean_df)} filas…")
        embs = generate_embeddings(
            texts=clean_df["frase"].tolist(),
            model_name=args.model,
            batch_size=args.batch_size,
            l2_normalize=args.l2
        )
        print(f"[OK] Dimensiones de embeddings: {embs.shape}")
        np.save(args.emb_npy, embs)
        print(f"[OK] Guardado binario: {args.emb_npy}")
        if not args.no_csv_emb:
            out_df = clean_df.copy()
            out_df["embedding"] = [x.tolist() for x in embs]
            out_df.to_csv(args.emb_csv, index=False)
            print(f"[OK] Guardado CSV (pesado): {args.emb_csv}")

def cargar_datos_supervisado_desde_pca(
    path_pca: str = "user_embeddings_PCA.csv"
):
    """
    Carga user_embeddings_PCA.csv y produce X, y listos para un modelo supervisado.

    El CSV debe tener:
        - 'User'
        - 'frase'
        - 'Label'
        - 'embedding' (lista como texto)
    """
    print(f"[INFO] Cargando dataset supervisado desde PCA: {path_pca}")
    df = pd.read_csv(path_pca)

    # Verificaciones
    for col in ["User", "frase", "Label", "embedding"]:
        if col not in df.columns:
            raise ValueError(
                f"Falta la columna '{col}' en {path_pca}. Columnas disponibles: {df.columns.tolist()}"
            )

    # Convertir embedding string → lista de floats
    print("[INFO] Parseando columna 'embedding' (string -> lista de floats)...")
    df["embedding"] = df["embedding"].apply(ast.literal_eval)

    # Expandir embedding en columnas numéricas
    print("[INFO] Expandiendo embeddings...")
    X = pd.DataFrame(df["embedding"].tolist())

    # Target
    y = df["Label"].astype(str)

    print(f"[OK] Datos supervisados listos: X={X.shape}, y={y.shape}")
    print(f"     Clases encontradas: {sorted(y.unique())}")

    return X, y, df



if __name__ == "__main__":
    main()

