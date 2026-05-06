import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — hanya untuk hal yang tidak bisa dilakukan Streamlit secara native
st.markdown("""
<style>
    /* Sidebar lebih lebar sedikit */
    [data-testid="stSidebar"] { min-width: 280px; }

    /* Metric card accent color */
    [data-testid="stMetric"] {
        background: #f8f7ff;
        border: 1px solid #e5e3f5;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* Tombol utama lebih menonjol */
    div.stButton > button[kind="primary"] {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        background-color: #3C3489;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# 2. MODEL PIPELINE (OOP)
# ─────────────────────────────────────────
class StudentGraduationPredictor:
    """Pipeline Machine Learning untuk prediksi kelulusan mahasiswa."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = DecisionTreeClassifier(criterion="gini", random_state=42)
        self.encoders: dict = {}
        self.feature_names: list = []
        self.target_name = "Kelulusan"
        self.classes_: list = []
        self.df_original: pd.DataFrame | None = None
        self.accuracy: float = 0.0

    def load_and_preprocess(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        self.df_original = df.copy()
        self.feature_names = [c for c in df.columns if c != self.target_name]

        df_encoded = df.copy()
        for col in df.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
            self.encoders[col] = le

        self.classes_ = list(self.encoders[self.target_name].classes_)

        X = df_encoded.drop(self.target_name, axis=1)
        y = df_encoded[self.target_name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return self.df_original

    def train_and_evaluate(self) -> tuple[any, float]:
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        return cm, self.accuracy

    def get_viz_figure(self, cm) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=130)
        fig.patch.set_facecolor("#fafaf8")

        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=self.classes_,
            filled=True,
            rounded=True,
            ax=axes[0],
            fontsize=10,
        )
        axes[0].set_title("Arsitektur Decision Tree", fontsize=14, fontweight="bold", pad=12)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes_)
        disp.plot(cmap=plt.cm.Blues, ax=axes[1], colorbar=False)
        axes[1].set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
        axes[1].set_xlabel("Prediksi", fontsize=11)
        axes[1].set_ylabel("Aktual", fontsize=11)

        plt.tight_layout(pad=2)
        return fig

    def predict(self, new_data: dict) -> str:
        df_new = pd.DataFrame([new_data])
        df_enc = df_new.copy()
        for col in self.feature_names:
            df_enc[col] = self.encoders[col].transform(df_new[col])
        pred = self.model.predict(df_enc)
        return self.encoders[self.target_name].inverse_transform(pred)[0]


# ─────────────────────────────────────────
# 3. LOAD & CACHE MODEL
# ─────────────────────────────────────────
DATA_PATH = "data_mahasiswa.csv"   # ← ganti path sesuai lokasi file kamu


@st.cache_resource(show_spinner="Melatih model, mohon tunggu...")
def load_model():
    pipeline = StudentGraduationPredictor(DATA_PATH)
    pipeline.load_and_preprocess()
    cm, acc = pipeline.train_and_evaluate()
    return pipeline, cm, acc


# ─────────────────────────────────────────
# 4. MAIN UI
# ─────────────────────────────────────────
def main():
    # ── Header ──────────────────────────
    st.title("🎓 Sistem Prediksi Kelulusan Mahasiswa")
    st.caption(
        "Menggunakan algoritma **Decision Tree (Gini)** untuk memprediksi "
        "ketepatan waktu kelulusan berdasarkan data latar belakang mahasiswa."
    )
    st.divider()

    # ── Load model ───────────────────────
    try:
        pipeline, conf_matrix, accuracy = load_model()
    except FileNotFoundError:
        st.error(
            f"⚠️ File `{DATA_PATH}` tidak ditemukan. "
            "Pastikan file CSV berada di folder yang sama dengan `app.py`."
        )
        st.stop()

    # ── Metric cards (ringkasan cepat) ──
    total_data = len(pipeline.df_original)
    n_train = len(pipeline.X_train)
    n_test = len(pipeline.X_test)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data", f"{total_data} mahasiswa")
    c2.metric("Data Training", f"{n_train} ({int(n_train/total_data*100)}%)")
    c3.metric("Data Testing", f"{n_test} ({int(n_test/total_data*100)}%)")
    c4.metric("Akurasi Model", f"{accuracy:.1%}")

    st.divider()

    # ── Tabs ─────────────────────────────
    tab_pred, tab_viz, tab_data = st.tabs([
        "Prediksi Mahasiswa Baru",
        "Visualisasi Model",
        "Dataset",
    ])

    # TAB 1 — PREDIKSI
    with tab_pred:
        col_form, col_result = st.columns([1, 1.4], gap="large")

        with col_form:
            st.subheader("Input Data Mahasiswa")
            with st.form("form_prediksi"):
                jk     = st.selectbox("Jenis Kelamin",      ["Laki-laki", "Perempuan"])
                asal   = st.selectbox("Asal Sekolah",       ["SMA", "SMK"])
                nikah  = st.selectbox("Status Pernikahan",  ["Belum", "Sudah"])
                ukuran = st.selectbox("Ukuran Program",     ["Reguler", "Ekstensi"])
                submit = st.form_submit_button("Prediksi Sekarang", type="primary")

        with col_result:
            st.subheader("Hasil Prediksi")

            if submit:
                input_data = {
                    "Jenis Kelamin": jk,
                    "Asal SMA":      asal,
                    "Nikah":         nikah,
                    "UkuranProgram": ukuran,
                }

                # Tabel ringkasan input
                st.dataframe(
                    pd.DataFrame([input_data]).rename(columns={
                        "Jenis Kelamin": "Jenis Kelamin",
                        "Asal SMA":      "Asal Sekolah",
                        "Nikah":         "Status Nikah",
                        "UkuranProgram": "Program",
                    }),
                    hide_index=True,
                    use_container_width=True,
                )

                hasil = pipeline.predict(input_data)

                if hasil == "Tepat":
                    st.success(f"###Lulus TEPAT WAKTU")
                    st.info(
                        "Berdasarkan data yang dimasukkan, mahasiswa ini diprediksi "
                        "akan menyelesaikan studi sesuai jadwal yang ditetapkan."
                    )
                else:
                    st.warning(f"### ⚠️ Berisiko TIDAK TEPAT WAKTU")
                    st.info(
                        "Berdasarkan data yang dimasukkan, mahasiswa ini diprediksi "
                        "berisiko mengalami keterlambatan dalam penyelesaian studi."
                    )
            else:
                st.info("Isi formulir di sebelah kiri, lalu klik **Prediksi Sekarang**.")

    # TAB 2 — VISUALISASI
    with tab_viz:
        st.subheader("Arsitektur Pohon Keputusan & Confusion Matrix")
        st.caption(
            "Decision Tree di bawah menggambarkan aturan keputusan yang dipelajari model. "
            "Confusion Matrix menunjukkan performa prediksi pada data testing."
        )
        fig = pipeline.get_viz_figure(conf_matrix)
        st.pyplot(fig, use_container_width=True)

   # TAB 3 — DATASET
    with tab_data:
        st.subheader("Dataset Mahasiswa")

        # Filter sederhana
        filter_col, _ = st.columns([1, 2])
        with filter_col:
            filter_kelulusan = st.selectbox(
                "Filter Kelulusan", ["Semua", "Tepat", "Tidak Tepat"]
            )

        df_show = pipeline.df_original.copy()
        if filter_kelulusan != "Semua":
            keyword = "Tepat" if filter_kelulusan == "Tepat" else "Tidak"
            df_show = df_show[df_show["Kelulusan"].str.contains(keyword)]

        st.dataframe(df_show, use_container_width=True, hide_index=True)
        st.caption(f"Menampilkan {len(df_show)} dari {total_data} data.")


if __name__ == "__main__":
    main()