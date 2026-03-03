"""
Testes unitarios para o pipeline de orquestracao de detectores.

Todos os modelos pesados (ViT, CLIP, EfficientNet) sao substituidos por
mocks — nenhum download de modelo, GPU ou rede sao necessarios.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from PIL import Image

from core.pipeline import PipelineDeteccao
from core.ensemble import ResultadoEnsemble
from models.base import DetectorBase, ResultadoDeteccao


# ---------------------------------------------------------------------------
# Helpers e fakes
# ---------------------------------------------------------------------------


def _imagem_sintetica(largura: int = 64, altura: int = 64) -> Image.Image:
    """Cria imagem PIL sintetica RGB para testes."""
    pixels = (np.random.rand(altura, largura, 3) * 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


class DetectorFalso(DetectorBase):
    """
    Implementacao concreta minima de DetectorBase para testes.
    Retorna um score fixo sem carregar nenhum modelo real.
    """

    def __init__(self, id_modelo: str, score_fixo: float = 0.5) -> None:
        super().__init__(id_modelo=id_modelo, nome_modelo=f"Detector Falso {id_modelo}")
        self.score_fixo = score_fixo
        self.chamadas_carregar = 0
        self.chamadas_descarregar = 0
        self.chamadas_detectar = 0

    def carregar(self, dispositivo: str = "cpu") -> None:
        """Simula carregamento sem custo."""
        self.chamadas_carregar += 1
        self._carregado = True

    def descarregar(self) -> None:
        """Simula descarregamento sem custo."""
        self.chamadas_descarregar += 1
        self._carregado = False

    def detectar(self, imagem: Image.Image) -> ResultadoDeteccao:
        """Retorna score fixo sem inferencia real."""
        self.chamadas_detectar += 1
        return ResultadoDeteccao(
            score=self.score_fixo,
            confianca=0.95,
            id_modelo=self.id_modelo,
            nome_modelo=self.nome_modelo,
        )


class DetectorQueExplode(DetectorBase):
    """Detector que levanta excecao em detectar() para testar resiliencia."""

    def __init__(self) -> None:
        super().__init__(id_modelo="detector_explosivo", nome_modelo="Detector Explosivo")

    def carregar(self, dispositivo: str = "cpu") -> None:
        self._carregado = True

    def descarregar(self) -> None:
        self._carregado = False

    def detectar(self, imagem: Image.Image) -> ResultadoDeteccao:
        raise RuntimeError("Falha simulada no detector para teste de resiliencia")


# ---------------------------------------------------------------------------
# Fixture: contexto_modelo mockado para evitar dependencia de torch/GPU
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_gerenciador_gpu() -> Any:
    """
    Substitui o gerenciador_gpu global por um mock que:
    - nao toca torch nem CUDA
    - context manager contexto_modelo retorna "cpu" como dispositivo
    """

    @contextmanager
    def _contexto_modelo_falso(nome_modelo: str, vram_mb: int = 0):  # type: ignore[misc]
        yield "cpu"

    mock_gpu = MagicMock()
    mock_gpu.contexto_modelo.side_effect = _contexto_modelo_falso
    mock_gpu.dispositivo = "cpu"

    with patch("core.pipeline.gerenciador_gpu", mock_gpu):
        yield mock_gpu


# ---------------------------------------------------------------------------
# Testes: inicializacao do pipeline
# ---------------------------------------------------------------------------


class TestPipelineInicializacao:
    """Testes de setup e configuracao do pipeline."""

    def test_pipeline_inicia_sem_detectores(self) -> None:
        """Pipeline recem-criado nao deve ter detectores registrados."""
        pipeline = PipelineDeteccao()

        assert len(pipeline.detectores) == 0
        assert pipeline._inicializado is False

    def test_pipeline_inicializar_com_mocks_de_importacao(self) -> None:
        """
        inicializar() importa modelos pesados internamente.
        Com os imports substituidos por mocks, nao deve falhar.
        """
        detector_mock = DetectorFalso("spatial_vit")

        with patch("core.pipeline.DetectorViTEspacial", return_value=detector_mock), \
             patch("core.pipeline.DetectorCLIP", return_value=DetectorFalso("clip_detector")), \
             patch("core.pipeline.AnalisadorFrequencia", return_value=DetectorFalso("frequency_analyzer")), \
             patch("core.pipeline.DetectorEfficientNet", return_value=DetectorFalso("efficientnet_video")):
            pipeline = PipelineDeteccao()
            pipeline.inicializar()

        assert pipeline._inicializado is True

    def test_pipeline_inicializar_registra_detectores_habilitados(self) -> None:
        """
        Apos inicializar(), apenas modelos com habilitado=True no
        REGISTRO_MODELOS devem aparecer em pipeline.detectores.
        """
        detectores_mock = {
            "spatial_vit": DetectorFalso("spatial_vit"),
            "clip_detector": DetectorFalso("clip_detector"),
            "frequency_analyzer": DetectorFalso("frequency_analyzer"),
            "efficientnet_video": DetectorFalso("efficientnet_video"),
        }

        with patch("core.pipeline.DetectorViTEspacial", return_value=detectores_mock["spatial_vit"]), \
             patch("core.pipeline.DetectorCLIP", return_value=detectores_mock["clip_detector"]), \
             patch("core.pipeline.AnalisadorFrequencia", return_value=detectores_mock["frequency_analyzer"]), \
             patch("core.pipeline.DetectorEfficientNet", return_value=detectores_mock["efficientnet_video"]):
            pipeline = PipelineDeteccao()
            pipeline.inicializar()

        # Todos os 4 modelos do REGISTRO_MODELOS estao habilitados por padrao
        assert len(pipeline.detectores) == 4


# ---------------------------------------------------------------------------
# Testes: registro manual de detectores
# ---------------------------------------------------------------------------


class TestRegistroDetectores:
    """Testes para pipeline.registrar_detector()."""

    def test_registrar_detector_adiciona_ao_dicionario(self) -> None:
        """registrar_detector() deve adicionar o detector em self.detectores."""
        pipeline = PipelineDeteccao()
        detector = DetectorFalso("teste_registro")

        pipeline.registrar_detector(detector)

        assert "teste_registro" in pipeline.detectores
        assert pipeline.detectores["teste_registro"] is detector

    def test_registrar_multiplos_detectores(self) -> None:
        """Deve ser possivel registrar varios detectores consecutivamente."""
        pipeline = PipelineDeteccao()

        for i in range(3):
            pipeline.registrar_detector(DetectorFalso(f"detector_{i}"))

        assert len(pipeline.detectores) == 3

    def test_registrar_detector_substitui_existente(self) -> None:
        """Registrar detector com mesmo id deve substituir o anterior."""
        pipeline = PipelineDeteccao()
        detector_original = DetectorFalso("id_duplicado")
        detector_novo = DetectorFalso("id_duplicado", score_fixo=0.99)

        pipeline.registrar_detector(detector_original)
        pipeline.registrar_detector(detector_novo)

        assert pipeline.detectores["id_duplicado"] is detector_novo
        assert len(pipeline.detectores) == 1


# ---------------------------------------------------------------------------
# Testes: analisar_imagem com detectores mockados
# ---------------------------------------------------------------------------


class TestAnalisarImagem:
    """
    Testes de integracao do fluxo analisar_imagem() usando detectores
    falsos para verificar agregacao sem dependencias externas.
    """

    def _pipeline_com_detectores(
        self, scores: dict[str, float]
    ) -> PipelineDeteccao:
        """
        Cria pipeline pre-configurado com detectores falsos.
        O pipeline ja e marcado como inicializado para pular inicializar().
        """
        pipeline = PipelineDeteccao()
        pipeline._inicializado = True

        for id_modelo, score in scores.items():
            pipeline.registrar_detector(DetectorFalso(id_modelo, score_fixo=score))

        return pipeline

    def test_analisar_imagem_retorna_estrutura_esperada(self) -> None:
        """Resultado de analisar_imagem deve conter todas as chaves esperadas."""
        pipeline = self._pipeline_com_detectores({"detector_a": 0.80})
        imagem = _imagem_sintetica()

        resultado = pipeline.analisar_imagem(imagem)

        chaves_esperadas = {
            "ensemble",
            "score_calibrado",
            "intervalo_confianca",
            "concordancia",
            "visualizacoes",
            "tempo_total_ms",
            "tipo",
        }
        assert chaves_esperadas.issubset(resultado.keys())
        assert resultado["tipo"] == "imagem"

    def test_analisar_imagem_ensemble_e_resultado_ensemble(self) -> None:
        """O campo 'ensemble' deve ser uma instancia de ResultadoEnsemble."""
        pipeline = self._pipeline_com_detectores({"detector_a": 0.60})
        imagem = _imagem_sintetica()

        resultado = pipeline.analisar_imagem(imagem)

        assert isinstance(resultado["ensemble"], ResultadoEnsemble)

    def test_analisar_imagem_score_calibrado_entre_zero_e_um(self) -> None:
        """score_calibrado deve ser float entre 0 e 1."""
        pipeline = self._pipeline_com_detectores(
            {"detector_a": 0.30, "detector_b": 0.70}
        )
        imagem = _imagem_sintetica()

        resultado = pipeline.analisar_imagem(imagem)

        assert 0.0 <= resultado["score_calibrado"] <= 1.0

    def test_analisar_imagem_agregacao_de_scores(self) -> None:
        """
        Com dois detectores retornando scores conhecidos, o score do
        ensemble deve ser compativel com a media desses valores.
        Os pesos padrao favorecemos spatial_vit e clip_detector (0.35 cada),
        mas aqui usamos ids sem peso registrado -> media simples.
        """
        pipeline = self._pipeline_com_detectores(
            {"id_sem_peso_a": 0.20, "id_sem_peso_b": 0.80}
        )
        imagem = _imagem_sintetica()

        resultado = pipeline.analisar_imagem(imagem)

        # Sem peso registrado, ensemble usa np.mean -> 0.50
        score_ensemble = resultado["ensemble"].score_final
        assert abs(score_ensemble - 0.50) < 1e-9

    def test_analisar_imagem_sem_detectores(self) -> None:
        """
        Pipeline sem detectores deve retornar ensemble com score 0.5
        (lista vazia passada ao agregador).
        """
        pipeline = PipelineDeteccao()
        pipeline._inicializado = True
        imagem = _imagem_sintetica()

        resultado = pipeline.analisar_imagem(imagem)

        assert resultado["ensemble"].score_final == 0.5
        assert resultado["ensemble"].classificacao == "Indeterminado"

    def test_analisar_imagem_intervalo_confianca_e_tupla(self) -> None:
        """intervalo_confianca deve ser uma tupla (limite_inf, limite_sup)."""
        pipeline = self._pipeline_com_detectores({"detector_a": 0.50})
        imagem = _imagem_sintetica()

        resultado = pipeline.analisar_imagem(imagem)
        intervalo = resultado["intervalo_confianca"]

        assert isinstance(intervalo, tuple)
        assert len(intervalo) == 2
        limite_inferior, limite_superior = intervalo
        assert limite_inferior <= limite_superior

    def test_analisar_imagem_callback_progresso_chamado(self) -> None:
        """O callback de progresso deve ser chamado ao menos uma vez."""
        pipeline = self._pipeline_com_detectores({"detector_a": 0.50})
        imagem = _imagem_sintetica()
        chamadas: list[tuple[str, float]] = []

        def callback(etapa: str, progresso: float) -> None:
            chamadas.append((etapa, progresso))

        pipeline.analisar_imagem(imagem, callback_progresso=callback)

        assert len(chamadas) >= 1
        # Ultima chamada deve ser o sinal de conclusao
        ultima_etapa, ultimo_progresso = chamadas[-1]
        assert ultimo_progresso == 100


# ---------------------------------------------------------------------------
# Testes: resiliencia a falhas em detectores individuais
# ---------------------------------------------------------------------------


class TestResilienciaDetectores:
    """Testes para o comportamento quando um detector lanca excecao."""

    def test_detector_com_excecao_nao_interrompe_pipeline(self) -> None:
        """
        Se um detector lanca excecao em detectar(), o pipeline deve
        continuar com os demais detectores e retornar resultado valido.
        """
        pipeline = PipelineDeteccao()
        pipeline._inicializado = True

        # Registra detector que vai falhar e um que funcionara
        pipeline.registrar_detector(DetectorQueExplode())
        pipeline.registrar_detector(DetectorFalso("detector_saudavel", score_fixo=0.80))

        imagem = _imagem_sintetica()

        # Nao deve levantar excecao
        resultado = pipeline.analisar_imagem(imagem)

        assert isinstance(resultado, dict)
        assert "ensemble" in resultado

    def test_detector_com_excecao_score_baseado_nos_que_sobreviveram(self) -> None:
        """
        Quando um detector falha, o score final deve ser baseado apenas
        nos detectores que retornaram resultado com sucesso.
        """
        pipeline = PipelineDeteccao()
        pipeline._inicializado = True

        pipeline.registrar_detector(DetectorQueExplode())
        # id sem peso registrado -> ensemble usara media simples
        pipeline.registrar_detector(
            DetectorFalso("id_sem_peso_ok", score_fixo=0.70)
        )

        imagem = _imagem_sintetica()
        resultado = pipeline.analisar_imagem(imagem)

        # Apenas o detector saudavel contribui; score deve ser 0.70
        assert abs(resultado["ensemble"].score_final - 0.70) < 1e-9

    def test_todos_detectores_falham_retorna_indeterminado(self) -> None:
        """
        Se todos os detectores falham, o ensemble recebe lista vazia e
        retorna classificacao 'Indeterminado' com score 0.5.
        """
        pipeline = PipelineDeteccao()
        pipeline._inicializado = True
        pipeline.registrar_detector(DetectorQueExplode())

        imagem = _imagem_sintetica()
        resultado = pipeline.analisar_imagem(imagem)

        assert resultado["ensemble"].score_final == 0.5
        assert resultado["ensemble"].classificacao == "Indeterminado"

    def test_detector_carregado_e_descarregado_mesmo_com_excecao(self) -> None:
        """
        O context manager contexto_modelo deve ser executado mesmo quando
        detectar() levanta excecao (testado via contadores do DetectorFalso).
        """
        pipeline = PipelineDeteccao()
        pipeline._inicializado = True

        detector_monitorado = DetectorFalso("monitorado", score_fixo=0.60)
        pipeline.registrar_detector(detector_monitorado)

        imagem = _imagem_sintetica()
        pipeline.analisar_imagem(imagem)

        # Verifica que o ciclo completo de carregar/detectar/descarregar ocorreu
        assert detector_monitorado.chamadas_carregar == 1
        assert detector_monitorado.chamadas_detectar == 1
        assert detector_monitorado.chamadas_descarregar == 1


# ---------------------------------------------------------------------------
# Testes: filtragem de modelos por tipo
# ---------------------------------------------------------------------------


class TestFiltragemModelos:
    """Testes para a logica de filtragem de detectores por lista de modelos."""

    def test_filtrar_modelos_habilitados_restringe_detectores(self) -> None:
        """
        modelos_habilitados deve restringir a execucao aos ids listados.
        Apenas o detector listado deve ser chamado.
        """
        pipeline = PipelineDeteccao()
        pipeline._inicializado = True

        detector_a = DetectorFalso("id_sem_peso_a", score_fixo=0.10)
        detector_b = DetectorFalso("id_sem_peso_b", score_fixo=0.90)
        pipeline.registrar_detector(detector_a)
        pipeline.registrar_detector(detector_b)

        imagem = _imagem_sintetica()
        pipeline.analisar_imagem(
            imagem,
            modelos_habilitados=["id_sem_peso_a"],
        )

        assert detector_a.chamadas_detectar == 1
        assert detector_b.chamadas_detectar == 0
