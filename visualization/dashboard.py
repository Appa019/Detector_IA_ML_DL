"""
Dashboard principal do Detector de Conteudo Gerado por IA.

Gerencia o layout completo da interface Streamlit, incluindo sidebar,
upload de arquivos e exibicao de resultados para imagens e videos.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

from config.model_registry import REGISTRO_MODELOS
from config.settings import CONFIG_APP, CONFIG_ENSEMBLE
from core.pipeline import PipelineDeteccao
from utils.gpu_manager import gerenciador_gpu

# Importacoes de visualizacao criadas por outro agente.
# Caso ainda nao existam, o dashboard captura o ImportError com graciosidade.
try:
    from visualization.charts import (
        criar_gauge_confianca,
        criar_barras_modelos,
        criar_distribuicao_scores,
        criar_grafico_concordancia,
        criar_tabela_metricas,
    )
    _CHARTS_DISPONIVEL = True
except ImportError:
    _CHARTS_DISPONIVEL = False

try:
    from visualization.heatmaps import (
        criar_visualizacao_gradcam,
        criar_espectro_frequencia,
        criar_histograma_rgb,
        criar_noise_print,
    )
    _HEATMAPS_DISPONIVEL = True
except ImportError:
    _HEATMAPS_DISPONIVEL = False

try:
    from visualization.heatmaps import VisualizadorMapasCalor
    _HEATMAPS_CLASSE_DISPONIVEL = True
except ImportError:
    _HEATMAPS_CLASSE_DISPONIVEL = False

try:
    from visualization.video_timeline import (
        criar_timeline_scores,
        criar_grade_frames_suspeitos,
    )
    _TIMELINE_DISPONIVEL = True
except ImportError:
    _TIMELINE_DISPONIVEL = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache do pipeline — inicializado uma unica vez por sessao do servidor
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Carregando modelos de deteccao...")
def _carregar_pipeline() -> PipelineDeteccao:
    """
    Carrega e inicializa o pipeline de deteccao.

    Decorado com st.cache_resource para que a inicializacao custosa
    (download e carregamento de pesos) ocorra apenas uma vez.

    Returns:
        Instancia inicializada de PipelineDeteccao.
    """
    pipeline = PipelineDeteccao()
    try:
        pipeline.inicializar()
    except Exception as excecao:
        logger.error(f"Falha ao inicializar pipeline: {excecao}")
        # Retorna pipeline parcialmente inicializado; analisar_imagem
        # tentara inicializar novamente antes de usar.
    return pipeline


# ---------------------------------------------------------------------------
# Classe principal do dashboard
# ---------------------------------------------------------------------------

class DashboardDeteccao:
    """
    Gerencia o layout e o fluxo de interacao do dashboard Streamlit.

    Responsabilidades:
    - Configurar a pagina (titulo, layout).
    - Renderizar a sidebar com info de GPU e configuracoes de modelos.
    - Orquestrar o upload e o processamento de arquivos.
    - Exibir resultados de imagens e videos com graficos interativos.
    """

    # Chaves de estado da sessao
    _CHAVE_RESULTADO = "resultado_atual"
    _CHAVE_ARQUIVO_NOME = "nome_arquivo_atual"
    _CHAVE_ARQUIVO_BYTES = "bytes_arquivo_atual"
    _CHAVE_TIPO_ARQUIVO = "tipo_arquivo_atual"

    def __init__(self) -> None:
        """Inicializa o pipeline e prepara o estado da sessao."""
        self._pipeline: PipelineDeteccao = _carregar_pipeline()
        self._garantir_estado_sessao()

    # ------------------------------------------------------------------
    # Inicializacao de estado
    # ------------------------------------------------------------------

    def _garantir_estado_sessao(self) -> None:
        """Cria chaves de session_state que ainda nao existem."""
        chaves_padrao: dict[str, object] = {
            self._CHAVE_RESULTADO: None,
            self._CHAVE_ARQUIVO_NOME: None,
            self._CHAVE_ARQUIVO_BYTES: None,
            self._CHAVE_TIPO_ARQUIVO: None,
        }
        for chave, valor_padrao in chaves_padrao.items():
            if chave not in st.session_state:
                st.session_state[chave] = valor_padrao

    # ------------------------------------------------------------------
    # Metodo principal de renderizacao
    # ------------------------------------------------------------------

    def renderizar(self) -> None:
        """
        Ponto de entrada da renderizacao completa do dashboard.

        Configura a pagina, renderiza a sidebar e a area principal
        com upload e resultados.
        """
        st.set_page_config(
            page_title=CONFIG_APP.titulo,
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                "About": (
                    f"**{CONFIG_APP.titulo}**\n\n"
                    f"{CONFIG_APP.descricao}"
                ),
            },
        )

        # Sidebar e modelos habilitados
        modelos_habilitados = self._renderizar_sidebar()

        # Area principal
        self._renderizar_area_principal(modelos_habilitados)

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    def _renderizar_sidebar(self) -> list[str]:
        """
        Renderiza a barra lateral com informacoes de GPU, toggles de modelos
        e sliders de threshold.

        Returns:
            Lista de IDs de modelos habilitados pelo usuario.
        """
        with st.sidebar:
            st.title("Configuracoes")
            st.divider()

            # --- Informacoes de GPU ---
            self._renderizar_info_gpu()
            st.divider()

            # --- Selecao de modelos ---
            st.subheader("Modelos")
            modelos_habilitados = self._renderizar_toggles_modelos()
            st.divider()

            # --- Thresholds ---
            self._renderizar_sliders_threshold()
            st.divider()

            # --- Sobre ---
            self._renderizar_sobre()

        return modelos_habilitados

    def _renderizar_info_gpu(self) -> None:
        """Exibe informacoes de GPU e barra de uso de VRAM na sidebar."""
        st.subheader("Hardware")

        try:
            info_gpu = gerenciador_gpu.obter_info()
        except Exception as excecao:
            logger.warning(f"Nao foi possivel obter info da GPU: {excecao}")
            st.warning("Informacoes de GPU indisponiveis.")
            return

        if info_gpu["disponivel"]:
            st.success(f"GPU: {info_gpu['nome']}")

            vram_total: float = info_gpu["vram_total_mb"]
            vram_usada: float = info_gpu["vram_usada_mb"]
            vram_livre: float = info_gpu["vram_livre_mb"]

            if vram_total > 0:
                fracao_usada = vram_usada / vram_total
                st.caption("Uso de VRAM")
                st.progress(
                    value=min(fracao_usada, 1.0),
                    text=f"{vram_usada:.0f} MB / {vram_total:.0f} MB",
                )

            col_usada, col_livre = st.columns(2)
            with col_usada:
                st.metric("Usada", f"{vram_usada:.0f} MB")
            with col_livre:
                st.metric("Livre", f"{vram_livre:.0f} MB")
        else:
            st.warning("GPU nao disponivel — usando CPU")
            st.caption("O processamento sera mais lento sem GPU.")

    def _renderizar_toggles_modelos(self) -> list[str]:
        """
        Exibe checkboxes para habilitar/desabilitar cada modelo do ensemble.

        Returns:
            Lista de IDs de modelos marcados como habilitados.
        """
        modelos_habilitados: list[str] = []

        for id_modelo, registro in REGISTRO_MODELOS.items():
            rotulo = f"{registro.nome_exibicao}"
            habilitado = st.checkbox(
                rotulo,
                value=registro.habilitado,
                key=f"modelo_{id_modelo}",
                help=(
                    f"Arquitetura: {registro.arquitetura}\n\n"
                    f"Papel: {registro.papel}\n\n"
                    f"VRAM estimada: {registro.vram_fp16_mb} MB"
                ),
            )
            if habilitado:
                modelos_habilitados.append(id_modelo)

        if not modelos_habilitados:
            st.warning("Selecione ao menos um modelo.")

        return modelos_habilitados

    def _renderizar_sliders_threshold(self) -> None:
        """
        Exibe sliders para ajuste dos thresholds de classificacao.

        Os valores ajustados sao armazenados no session_state e usados
        pelo ensemble na proxima analise.
        """
        st.subheader("Thresholds")
        st.caption("Ajuste os limites de classificacao:")

        thresholds = CONFIG_ENSEMBLE.thresholds

        st.session_state["threshold_provavelmente_real"] = st.slider(
            "Provavelmente Real (<)",
            min_value=0.05,
            max_value=0.45,
            value=st.session_state.get(
                "threshold_provavelmente_real",
                thresholds["provavelmente_real"],
            ),
            step=0.05,
            format="%.2f",
            help="Score abaixo deste valor e classificado como 'Provavelmente Real'.",
        )

        st.session_state["threshold_possivelmente_real"] = st.slider(
            "Possivelmente Real (<)",
            min_value=0.25,
            max_value=0.65,
            value=st.session_state.get(
                "threshold_possivelmente_real",
                thresholds["possivelmente_real"],
            ),
            step=0.05,
            format="%.2f",
            help="Score abaixo deste valor e classificado como 'Possivelmente Real'.",
        )

        st.session_state["threshold_possivelmente_ia"] = st.slider(
            "Possivelmente IA (<)",
            min_value=0.50,
            max_value=0.90,
            value=st.session_state.get(
                "threshold_possivelmente_ia",
                thresholds["possivelmente_ia"],
            ),
            step=0.05,
            format="%.2f",
            help="Score abaixo deste valor e classificado como 'Possivelmente IA'.",
        )

    def _renderizar_sobre(self) -> None:
        """Exibe secao 'Sobre' com descricao da aplicacao na sidebar."""
        with st.expander("Sobre"):
            st.markdown(
                f"""
                **{CONFIG_APP.titulo}**

                {CONFIG_APP.descricao}

                **Modelos disponíveis:**
                - ViT Espacial (Deep Fake Detector v2)
                - CLIP Detector
                - Analisador de Frequencia (FFT/DCT)
                - EfficientNet-B4

                **Limite de upload:** {CONFIG_APP.max_tamanho_upload_mb} MB

                ---
                *Desenvolvido com FastAPI + Streamlit*
                """
            )

    # ------------------------------------------------------------------
    # Area principal
    # ------------------------------------------------------------------

    def _renderizar_area_principal(self, modelos_habilitados: list[str]) -> None:
        """
        Renderiza o cabecalho, upload e resultados na area principal.

        Args:
            modelos_habilitados: IDs dos modelos selecionados na sidebar.
        """
        # Cabecalho
        st.title(f"🔍 {CONFIG_APP.titulo}")
        st.caption(CONFIG_APP.descricao)
        st.divider()

        # Upload
        arquivo_carregado, tipo_arquivo = self._renderizar_upload()

        if arquivo_carregado is not None:
            # Verifica se e um novo arquivo ou o mesmo ja processado
            novo_arquivo = (
                arquivo_carregado.name != st.session_state[self._CHAVE_ARQUIVO_NOME]
            )

            if novo_arquivo:
                # Limpa resultado anterior
                st.session_state[self._CHAVE_RESULTADO] = None
                st.session_state[self._CHAVE_ARQUIVO_NOME] = arquivo_carregado.name
                st.session_state[self._CHAVE_TIPO_ARQUIVO] = tipo_arquivo

                # Processa o arquivo
                self._processar_arquivo(
                    arquivo_carregado,
                    tipo_arquivo,
                    modelos_habilitados,
                )

        # Exibe resultado armazenado no estado da sessao
        resultado = st.session_state.get(self._CHAVE_RESULTADO)
        tipo_atual = st.session_state.get(self._CHAVE_TIPO_ARQUIVO)

        if resultado is not None:
            st.divider()
            if tipo_atual == "imagem":
                self._renderizar_resultado_imagem(resultado)
            elif tipo_atual == "video":
                self._renderizar_resultado_video(resultado)

        elif arquivo_carregado is None:
            # Estado inicial — sem arquivo carregado
            self._renderizar_estado_inicial()

    def _renderizar_estado_inicial(self) -> None:
        """Exibe mensagem de boas-vindas quando nenhum arquivo foi enviado."""
        st.info(
            "Envie uma imagem ou video para iniciar a analise. "
            "Use o seletor acima para escolher o arquivo.",
            icon="ℹ️",
        )
        st.markdown(
            """
            **Como funciona:**
            1. Faça upload de uma imagem (JPG, PNG, WEBP...) ou video (MP4, AVI, MOV...)
            2. O sistema analisara o arquivo com um ensemble de modelos de Deep Learning
            3. Voce recebera um score de probabilidade de conteudo gerado por IA
            4. Resultados detalhados incluem heatmaps, frequencias e estatisticas

            **Formatos suportados:**
            - Imagens: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`
            - Videos: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
            """
        )

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def _renderizar_upload(self) -> tuple[object | None, str | None]:
        """
        Renderiza a area de upload de arquivos.

        Returns:
            Tupla (arquivo_carregado, tipo_arquivo) onde tipo_arquivo e
            "imagem" ou "video", ou (None, None) se nenhum arquivo foi enviado.
        """
        # Tipos aceitos derivados da configuracao
        tipos_aceitos = list(CONFIG_APP.extensoes_imagem) + list(CONFIG_APP.extensoes_video)

        arquivo = st.file_uploader(
            label="Selecione uma imagem ou video",
            type=[ext.lstrip(".") for ext in tipos_aceitos],
            accept_multiple_files=False,
            help=(
                f"Tamanho maximo: {CONFIG_APP.max_tamanho_upload_mb} MB. "
                f"Imagens: {', '.join(CONFIG_APP.extensoes_imagem)}. "
                f"Videos: {', '.join(CONFIG_APP.extensoes_video)}."
            ),
        )

        if arquivo is None:
            return None, None

        # Determina o tipo pelo sufixo do nome
        sufixo = Path(arquivo.name).suffix.lower()

        if sufixo in CONFIG_APP.extensoes_imagem:
            tipo_arquivo = "imagem"
        elif sufixo in CONFIG_APP.extensoes_video:
            tipo_arquivo = "video"
        else:
            st.error(f"Formato '{sufixo}' nao suportado.")
            return None, None

        return arquivo, tipo_arquivo

    # ------------------------------------------------------------------
    # Processamento
    # ------------------------------------------------------------------

    def _processar_arquivo(
        self,
        arquivo: object,
        tipo_arquivo: str,
        modelos_habilitados: list[str],
    ) -> None:
        """
        Processa o arquivo enviado pelo usuario chamando o pipeline apropriado.

        Exibe barra de progresso durante a analise e armazena o resultado
        no session_state para exibicao posterior.

        Args:
            arquivo: Objeto UploadedFile do Streamlit.
            tipo_arquivo: "imagem" ou "video".
            modelos_habilitados: Lista de IDs de modelos a usar.
        """
        if not modelos_habilitados:
            st.error(
                "Nenhum modelo habilitado. Selecione ao menos um modelo na sidebar.",
                icon="⚠️",
            )
            return

        barra_progresso = st.progress(0, text="Iniciando analise...")
        area_status = st.empty()

        def _callback_progresso(etapa: str, percentual: float) -> None:
            """Atualiza barra de progresso durante a analise."""
            progresso_normalizado = min(int(percentual), 100) / 100
            barra_progresso.progress(
                progresso_normalizado,
                text=f"Analisando: {etapa}...",
            )

        try:
            if tipo_arquivo == "imagem":
                with st.spinner("Carregando imagem..."):
                    try:
                        imagem_pil = Image.open(arquivo).convert("RGB")
                    except Exception as excecao:
                        st.error(
                            f"Erro ao abrir a imagem: {excecao}. "
                            "Verifique se o arquivo nao esta corrompido.",
                            icon="❌",
                        )
                        barra_progresso.empty()
                        return

                area_status.info("Analisando imagem com ensemble de modelos...", icon="🔄")

                resultado = self._pipeline.analisar_imagem(
                    imagem=imagem_pil,
                    modelos_habilitados=modelos_habilitados,
                    callback_progresso=_callback_progresso,
                )
                # Guarda a imagem PIL para exibicao posterior
                resultado["_imagem_pil"] = imagem_pil

            elif tipo_arquivo == "video":
                # Salva em arquivo temporario para o pipeline de video
                with tempfile.NamedTemporaryFile(
                    suffix=Path(arquivo.name).suffix,
                    delete=False,
                ) as arq_temp:
                    arq_temp.write(arquivo.read())
                    caminho_temp = arq_temp.name

                area_status.info("Processando video (pode levar alguns minutos)...", icon="🔄")

                resultado = self._pipeline.analisar_video(
                    caminho_video=caminho_temp,
                    callback_progresso=_callback_progresso,
                )

                # Remove arquivo temporario
                try:
                    Path(caminho_temp).unlink()
                except OSError:
                    pass
            else:
                st.error("Tipo de arquivo desconhecido.", icon="❌")
                barra_progresso.empty()
                return

        except Exception as excecao:
            logger.exception(f"Erro durante a analise do arquivo '{arquivo.name}': {excecao}")
            st.error(
                f"Ocorreu um erro durante a analise: {excecao}. "
                "Verifique os logs para mais detalhes.",
                icon="❌",
            )
            barra_progresso.empty()
            area_status.empty()
            return

        # Armazena resultado e limpa elementos de progresso
        st.session_state[self._CHAVE_RESULTADO] = resultado
        barra_progresso.empty()
        area_status.empty()
        st.success("Analise concluida com sucesso!", icon="✅")

    # ------------------------------------------------------------------
    # Resultado — Imagem
    # ------------------------------------------------------------------

    def _renderizar_resultado_imagem(self, resultado: dict) -> None:
        """
        Exibe os resultados completos da analise de imagem.

        Layout em duas colunas:
        - Coluna 1 (1/3): imagem original + metadados basicos.
        - Coluna 2 (2/3): abas com graficos e analises detalhadas.

        Args:
            resultado: Dicionario retornado por PipelineDeteccao.analisar_imagem.
        """
        ensemble = resultado.get("ensemble")
        score_calibrado: float = resultado.get("score_calibrado", 0.5)
        intervalo: tuple = resultado.get("intervalo_confianca", (0.0, 1.0))
        concordancia: float = resultado.get("concordancia", 0.0)
        visualizacoes: dict = resultado.get("visualizacoes", {})
        tempo_ms: float = resultado.get("tempo_total_ms", 0.0)
        imagem_pil: Optional[Image.Image] = resultado.get("_imagem_pil")

        col_imagem, col_resultados = st.columns([1, 2])

        # --- Coluna esquerda: imagem + resumo ---
        with col_imagem:
            st.subheader("Imagem Analisada")

            if imagem_pil is not None:
                st.image(imagem_pil, use_container_width=True)
            else:
                st.caption("Imagem nao disponivel para exibicao.")

            if ensemble is not None:
                # Metrica principal em destaque
                st.metric(
                    label="Classificacao",
                    value=ensemble.classificacao,
                    help="Classificacao do ensemble de modelos.",
                )

                col_score, col_incert = st.columns(2)
                with col_score:
                    st.metric(
                        "Score IA",
                        f"{score_calibrado:.1%}",
                        help="0% = certamente real, 100% = certamente gerado por IA.",
                    )
                with col_incert:
                    st.metric(
                        "Incerteza",
                        f"{ensemble.incerteza:.1%}",
                        help="Desvio padrao entre os modelos do ensemble.",
                    )

                st.caption(
                    f"Intervalo de confianca: [{intervalo[0]:.1%}, {intervalo[1]:.1%}]"
                )
                st.caption(f"Tempo de analise: {tempo_ms:.0f} ms")

        # --- Coluna direita: abas de resultados ---
        with col_resultados:
            aba_resultado, aba_analise, aba_estatisticas, aba_metadados = st.tabs([
                "Resultado",
                "Analise Detalhada",
                "Estatisticas",
                "Metadados",
            ])

            with aba_resultado:
                self._renderizar_aba_resultado_imagem(
                    resultado=resultado,
                    score_calibrado=score_calibrado,
                    concordancia=concordancia,
                )

            with aba_analise:
                self._renderizar_aba_analise_imagem(
                    visualizacoes=visualizacoes,
                    imagem_pil=imagem_pil,
                )

            with aba_estatisticas:
                self._renderizar_aba_estatisticas_imagem(
                    resultado=resultado,
                    concordancia=concordancia,
                )

            with aba_metadados:
                self._renderizar_aba_metadados(resultado=resultado)

    def _renderizar_aba_resultado_imagem(
        self,
        resultado: dict,
        score_calibrado: float,
        concordancia: float,
    ) -> None:
        """
        Conteudo da aba 'Resultado': gauge de confianca, barras por modelo
        e indicador de incerteza.

        Args:
            resultado: Dicionario completo do resultado do pipeline.
            score_calibrado: Score final calibrado.
            concordancia: Percentual de concordancia entre os modelos.
        """
        ensemble = resultado.get("ensemble")
        if ensemble is None:
            st.warning("Resultado do ensemble nao disponivel.")
            return

        if _CHARTS_DISPONIVEL:
            # Gauge principal
            fig_gauge = criar_gauge_confianca(
                score=score_calibrado,
                classificacao=ensemble.classificacao,
                incerteza=ensemble.incerteza,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Barras por modelo
            if ensemble.scores_individuais:
                fig_barras = criar_barras_modelos(
                    scores_individuais=ensemble.scores_individuais,
                )
                st.plotly_chart(fig_barras, use_container_width=True)
        else:
            # Fallback sem Plotly
            st.warning("Modulo de graficos nao carregado. Exibindo valores numericos.")
            st.metric("Score Final", f"{score_calibrado:.1%}")
            if ensemble.scores_individuais:
                st.write("**Scores por modelo:**")
                for id_modelo, score in ensemble.scores_individuais.items():
                    st.progress(score, text=f"{id_modelo}: {score:.1%}")

        # Indicador de concordancia
        st.metric(
            "Concordancia entre modelos",
            f"{concordancia:.1%}",
            help="Percentual de concordancia na classificacao entre os modelos do ensemble.",
        )

    def _renderizar_aba_analise_imagem(
        self,
        visualizacoes: dict,
        imagem_pil: Optional[Image.Image],
    ) -> None:
        """
        Conteudo da aba 'Analise Detalhada': regioes suspeitas, ELA,
        consistencia de ruido, GradCAM, espectro FFT, histograma e noise print.

        Args:
            visualizacoes: Dicionario com arrays de visualizacao do pipeline.
            imagem_pil: Imagem PIL original para referencia.
        """
        import numpy as np  # noqa: PLC0415

        # --- Secao 1: Regioes Suspeitas (fusao) ---
        if _HEATMAPS_CLASSE_DISPONIVEL and imagem_pil is not None:
            st.markdown("### Regioes Suspeitas")
            try:
                viz = VisualizadorMapasCalor()
                imagem_np = np.array(imagem_pil)
                mapa_calor = _extrair_primeiro_mapa_calor(visualizacoes)
                fig_suspeitas = viz.criar_visualizacao_regioes_suspeitas(
                    imagem_np, mapa_calor_gradcam=mapa_calor
                )
                st.pyplot(fig_suspeitas)
                import matplotlib.pyplot as _plt  # noqa: PLC0415
                _plt.close(fig_suspeitas)
            except Exception as e:
                st.caption(f"Erro ao gerar mapa de regioes suspeitas: {e}")

            st.divider()

            # --- Secao 2: ELA e Consistencia de Ruido (2 colunas) ---
            col_ela, col_ruido = st.columns(2)

            with col_ela:
                st.markdown("**Error Level Analysis (ELA)**")
                try:
                    fig_ela = viz.criar_visualizacao_ela(imagem_np)
                    st.pyplot(fig_ela)
                    _plt.close(fig_ela)
                except Exception as e:
                    st.caption(f"Erro ao gerar ELA: {e}")

            with col_ruido:
                st.markdown("**Consistencia de Ruido**")
                try:
                    fig_ruido = viz.criar_visualizacao_consistencia_ruido(imagem_np)
                    st.pyplot(fig_ruido)
                    _plt.close(fig_ruido)
                except Exception as e:
                    st.caption(f"Erro ao gerar mapa de ruido: {e}")

            st.divider()

            # --- Secao 3: GradCAM e Espectro FFT (2 colunas) ---
            col_gradcam, col_fft = st.columns(2)

            with col_gradcam:
                st.markdown("**Mapa de Atencao (GradCAM)**")
                mapa_calor = _extrair_primeiro_mapa_calor(visualizacoes)
                if mapa_calor is not None:
                    try:
                        fig_gradcam = viz.criar_visualizacao_gradcam(imagem_np, mapa_calor)
                        st.pyplot(fig_gradcam)
                        _plt.close(fig_gradcam)
                    except Exception as e:
                        st.caption(f"Erro ao gerar GradCAM: {e}")
                else:
                    st.caption("Mapa de atencao nao disponivel para este modelo.")

            with col_fft:
                st.markdown("**Espectro de Frequencia (FFT)**")
                try:
                    fig_fft = viz.criar_visualizacao_espectro(imagem_np)
                    st.pyplot(fig_fft)
                    _plt.close(fig_fft)
                except Exception as e:
                    st.caption(f"Erro ao gerar espectro FFT: {e}")

            st.divider()

            # --- Secao 4: Histograma RGB e Noise Print ---
            col_hist, col_noise = st.columns(2)

            with col_hist:
                st.markdown("**Histograma RGB**")
                try:
                    fig_hist = viz.criar_visualizacao_histograma_rgb(imagem_np)
                    st.pyplot(fig_hist)
                    _plt.close(fig_hist)
                except Exception as e:
                    st.caption(f"Erro ao gerar histograma: {e}")

            with col_noise:
                st.markdown("**Noise Print**")
                try:
                    fig_noise = viz.criar_visualizacao_noise_print(imagem_np)
                    st.pyplot(fig_noise)
                    _plt.close(fig_noise)
                except Exception as e:
                    st.caption(f"Erro ao gerar noise print: {e}")

        elif _HEATMAPS_DISPONIVEL and imagem_pil is not None:
            # Fallback: usa funcoes standalone (sem novas visualizacoes)
            col_esq, col_dir = st.columns(2)

            with col_esq:
                st.markdown("**Mapa de Atencao (GradCAM)**")
                mapa_calor = _extrair_primeiro_mapa_calor(visualizacoes)
                if mapa_calor is not None:
                    fig_gradcam = criar_visualizacao_gradcam(
                        imagem=imagem_pil, mapa_calor=mapa_calor,
                    )
                    st.plotly_chart(fig_gradcam, use_container_width=True)
                else:
                    st.caption("Mapa de atencao nao disponivel para este modelo.")

                st.markdown("**Histograma RGB**")
                fig_hist = criar_histograma_rgb(imagem=imagem_pil)
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_dir:
                st.markdown("**Espectro de Frequencia (FFT)**")
                fig_fft = criar_espectro_frequencia(imagem=imagem_pil)
                st.plotly_chart(fig_fft, use_container_width=True)

                st.markdown("**Noise Print**")
                fig_noise = criar_noise_print(imagem=imagem_pil)
                st.plotly_chart(fig_noise, use_container_width=True)
        else:
            st.info(
                "Modulo de heatmaps nao carregado. "
                "As visualizacoes detalhadas serao exibidas quando disponivel.",
                icon="ℹ️",
            )

    def _renderizar_aba_estatisticas_imagem(
        self,
        resultado: dict,
        concordancia: float,
    ) -> None:
        """
        Conteudo da aba 'Estatisticas': grafico de concordancia, distribuicao
        de scores e tabela de metricas detalhadas.

        Args:
            resultado: Dicionario completo do resultado do pipeline.
            concordancia: Percentual de concordancia entre os modelos.
        """
        ensemble = resultado.get("ensemble")
        if ensemble is None:
            st.warning("Dados estatisticos nao disponiveis.")
            return

        if _CHARTS_DISPONIVEL:
            col_conc, col_dist = st.columns(2)

            with col_conc:
                fig_concordancia = criar_grafico_concordancia(
                    concordancia=concordancia,
                    scores_individuais=ensemble.scores_individuais,
                )
                st.plotly_chart(fig_concordancia, use_container_width=True)

            with col_dist:
                if ensemble.scores_individuais:
                    fig_dist = criar_distribuicao_scores(
                        scores=list(ensemble.scores_individuais.values()),
                        labels=list(ensemble.scores_individuais.keys()),
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

            st.markdown("**Metricas Detalhadas**")
            metricas = _montar_metricas_detalhadas(resultado)
            fig_tabela = criar_tabela_metricas(metricas=metricas)
            st.plotly_chart(fig_tabela, use_container_width=True)
        else:
            st.info("Modulo de graficos nao carregado.", icon="ℹ️")
            if ensemble.scores_individuais:
                st.json(ensemble.scores_individuais)

    def _renderizar_aba_metadados(self, resultado: dict) -> None:
        """
        Conteudo da aba 'Metadados': tabela com metadados EXIF e informacoes
        tecnicas do arquivo.

        Args:
            resultado: Dicionario completo do resultado do pipeline.
        """
        # Metadados EXIF podem ser retornados pelo pipeline ou pelo analysis layer
        metadados_exif: dict = resultado.get("metadados_exif", {})
        metadados_arquivo: dict = resultado.get("metadados_arquivo", {})

        if metadados_exif or metadados_arquivo:
            if metadados_exif:
                st.markdown("**Metadados EXIF**")
                _exibir_tabela_metadados(metadados_exif)

            if metadados_arquivo:
                st.markdown("**Informacoes do Arquivo**")
                _exibir_tabela_metadados(metadados_arquivo)
        else:
            st.info(
                "Nenhum metadado disponivel para este arquivo. "
                "Arquivos gerados por IA geralmente nao possuem metadados EXIF.",
                icon="ℹ️",
            )

    # ------------------------------------------------------------------
    # Resultado — Video
    # ------------------------------------------------------------------

    def _renderizar_resultado_video(self, resultado: dict) -> None:
        """
        Exibe os resultados completos da analise de video.

        Layout semelhante ao de imagem, com abas adicionais de timeline
        e frames suspeitos.

        Args:
            resultado: Dicionario retornado por PipelineDeteccao.analisar_video.
        """
        ensemble = resultado.get("ensemble")
        score_calibrado: float = resultado.get("score_calibrado", 0.5)
        intervalo: tuple = resultado.get("intervalo_confianca", (0.0, 1.0))
        timeline: list = resultado.get("timeline", [])
        frames_suspeitos: list = resultado.get("frames_suspeitos", [])
        total_frames: int = resultado.get("total_frames", 0)
        tempo_ms: float = resultado.get("tempo_total_ms", 0.0)
        erro: Optional[str] = resultado.get("erro")

        if erro:
            st.error(f"Erro no processamento do video: {erro}", icon="❌")
            return

        col_resumo, col_resultados = st.columns([1, 2])

        # --- Coluna esquerda: resumo ---
        with col_resumo:
            st.subheader("Resumo do Video")

            if ensemble is not None:
                st.metric(
                    label="Classificacao",
                    value=ensemble.classificacao,
                )

                col_score, col_incert = st.columns(2)
                with col_score:
                    st.metric("Score IA", f"{score_calibrado:.1%}")
                with col_incert:
                    st.metric("Incerteza", f"{ensemble.incerteza:.1%}")

                st.caption(
                    f"Intervalo de confianca: [{intervalo[0]:.1%}, {intervalo[1]:.1%}]"
                )

            st.metric("Frames Processados", total_frames)
            st.caption(f"Tempo de analise: {tempo_ms / 1000:.1f} s")

        # --- Coluna direita: abas ---
        with col_resultados:
            (
                aba_resultado,
                aba_timeline,
                aba_frames_suspeitos,
                aba_estatisticas,
                aba_metadados,
            ) = st.tabs([
                "Resultado",
                "Timeline",
                "Frames Suspeitos",
                "Estatisticas",
                "Metadados",
            ])

            with aba_resultado:
                self._renderizar_aba_resultado_video(
                    resultado=resultado,
                    score_calibrado=score_calibrado,
                )

            with aba_timeline:
                self._renderizar_aba_timeline(timeline=timeline)

            with aba_frames_suspeitos:
                self._renderizar_aba_frames_suspeitos(
                    frames_suspeitos=frames_suspeitos,
                )

            with aba_estatisticas:
                self._renderizar_aba_estatisticas_video(resultado=resultado)

            with aba_metadados:
                self._renderizar_aba_metadados(resultado=resultado)

    def _renderizar_aba_resultado_video(
        self,
        resultado: dict,
        score_calibrado: float,
    ) -> None:
        """
        Conteudo da aba 'Resultado' para video: gauge e barras de scores.

        Args:
            resultado: Dicionario completo do resultado do pipeline.
            score_calibrado: Score final calibrado.
        """
        ensemble = resultado.get("ensemble")
        if ensemble is None:
            st.warning("Resultado do ensemble nao disponivel.")
            return

        if _CHARTS_DISPONIVEL:
            fig_gauge = criar_gauge_confianca(
                score=score_calibrado,
                classificacao=ensemble.classificacao,
                incerteza=ensemble.incerteza,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            if ensemble.scores_individuais:
                fig_barras = criar_barras_modelos(
                    scores_individuais=ensemble.scores_individuais,
                )
                st.plotly_chart(fig_barras, use_container_width=True)
        else:
            st.metric("Score Final", f"{score_calibrado:.1%}")

    def _renderizar_aba_timeline(self, timeline: list[dict]) -> None:
        """
        Conteudo da aba 'Timeline': grafico frame-a-frame com scores ao longo
        do tempo.

        Args:
            timeline: Lista de dicts com 'indice_frame', 'score', 'classificacao'
                      e 'tem_rosto' para cada frame analisado.
        """
        if not timeline:
            st.info("Nenhum dado de timeline disponivel.", icon="ℹ️")
            return

        if _TIMELINE_DISPONIVEL:
            fig_timeline = criar_timeline_scores(dados_timeline=timeline)
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("Modulo de timeline nao carregado.", icon="ℹ️")
            # Fallback: tabela simples
            import pandas as pd  # noqa: PLC0415
            df = pd.DataFrame(timeline)
            st.dataframe(df, use_container_width=True)

    def _renderizar_aba_frames_suspeitos(
        self,
        frames_suspeitos: list[dict],
    ) -> None:
        """
        Conteudo da aba 'Frames Suspeitos': grade dos frames com maior score
        de IA detectado.

        Args:
            frames_suspeitos: Lista de dicts com informacoes dos frames mais
                              suspeitos ordenados por score decrescente.
        """
        if not frames_suspeitos:
            st.info("Nenhum frame suspeito identificado.", icon="ℹ️")
            return

        if _TIMELINE_DISPONIVEL:
            fig_grade = criar_grade_frames_suspeitos(
                frames_suspeitos=frames_suspeitos,
            )
            st.plotly_chart(fig_grade, use_container_width=True)
        else:
            st.info("Modulo de visualizacao de frames nao carregado.", icon="ℹ️")
            # Fallback: tabela simples
            st.markdown("**Frames com maior score de IA:**")
            for i, frame in enumerate(frames_suspeitos, start=1):
                score = frame.get("score", 0.0)
                indice = frame.get("indice_frame", "?")
                classificacao = frame.get("classificacao", "")
                st.write(f"{i}. Frame {indice} — Score: {score:.1%} — {classificacao}")

    def _renderizar_aba_estatisticas_video(self, resultado: dict) -> None:
        """
        Conteudo da aba 'Estatisticas' para video: distribuicao de scores
        por frame e metricas agregadas.

        Args:
            resultado: Dicionario completo do resultado do pipeline.
        """
        ensemble = resultado.get("ensemble")
        timeline: list = resultado.get("timeline", [])

        if _CHARTS_DISPONIVEL and timeline:
            scores_frames = [f["score"] for f in timeline]
            labels_frames = [f"Frame {f['indice_frame']}" for f in timeline]

            fig_dist = criar_distribuicao_scores(
                scores=scores_frames,
                labels=labels_frames,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        if ensemble is not None:
            st.markdown("**Metricas do Video**")
            metricas = _montar_metricas_detalhadas(resultado)
            if _CHARTS_DISPONIVEL:
                fig_tabela = criar_tabela_metricas(metricas=metricas)
                st.plotly_chart(fig_tabela, use_container_width=True)
            else:
                st.json(metricas)
        else:
            st.info("Dados estatisticos nao disponiveis.", icon="ℹ️")


# ---------------------------------------------------------------------------
# Funcoes auxiliares privadas ao modulo
# ---------------------------------------------------------------------------

def _extrair_primeiro_mapa_calor(visualizacoes: dict) -> object | None:
    """
    Extrai o primeiro mapa de calor disponivel no dicionario de visualizacoes.

    Args:
        visualizacoes: Dicionario retornado pelo pipeline com chaves
                       no formato 'heatmap_{id_modelo}'.

    Returns:
        Array numpy do mapa de calor, ou None se nao houver nenhum.
    """
    for chave, valor in visualizacoes.items():
        if chave.startswith("heatmap_") and valor is not None:
            return valor
    return None


def _exibir_tabela_metadados(metadados: dict) -> None:
    """
    Exibe um dicionario de metadados como tabela Streamlit formatada.

    Args:
        metadados: Dicionario chave-valor com os metadados a exibir.
    """
    linhas = [
        {"Campo": chave, "Valor": str(valor)}
        for chave, valor in metadados.items()
        if valor is not None
    ]
    if linhas:
        import pandas as pd  # noqa: PLC0415
        st.dataframe(
            pd.DataFrame(linhas),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Nenhum dado disponivel.")


def _montar_metricas_detalhadas(resultado: dict) -> dict:
    """
    Monta dicionario de metricas formatadas para exibicao na tabela.

    Args:
        resultado: Dicionario completo do resultado do pipeline.

    Returns:
        Dicionario com metricas formatadas como strings.
    """
    ensemble = resultado.get("ensemble")
    score_calibrado: float = resultado.get("score_calibrado", 0.5)
    intervalo: tuple = resultado.get("intervalo_confianca", (0.0, 1.0))
    concordancia: float = resultado.get("concordancia", 0.0)
    tempo_ms: float = resultado.get("tempo_total_ms", 0.0)

    metricas: dict[str, str] = {
        "Score Calibrado": f"{score_calibrado:.4f}",
        "Classificacao": ensemble.classificacao if ensemble else "N/A",
        "Incerteza": f"{ensemble.incerteza:.4f}" if ensemble else "N/A",
        "Intervalo de Confianca": (
            f"[{intervalo[0]:.4f}, {intervalo[1]:.4f}]"
        ),
        "Concordancia entre Modelos": f"{concordancia:.1%}",
        "Tempo de Analise": f"{tempo_ms:.0f} ms",
    }

    if ensemble and ensemble.scores_individuais:
        for id_modelo, score in ensemble.scores_individuais.items():
            metricas[f"Score — {id_modelo}"] = f"{score:.4f}"

    return metricas
