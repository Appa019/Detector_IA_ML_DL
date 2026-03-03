"""
Testes unitarios para modelos base, registro de modelos e calibracao.

Nenhum modelo e carregado de verdade — todos os testes usam mocks e
classes concretas minimas. Sem GPU, rede ou downloads necessarios.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from PIL import Image

from config.model_registry import (
    REGISTRO_MODELOS,
    RegistroModelo,
    calcular_vram_total,
    obter_modelo,
    obter_modelos_por_tipo,
)
from core.confidence import CalibradorConfianca
from models.base import DetectorBase, ResultadoDeteccao


# ---------------------------------------------------------------------------
# Implementacao concreta minima para testar DetectorBase
# ---------------------------------------------------------------------------


class DetectorConcreto(DetectorBase):
    """
    Subclasse concreta de DetectorBase para verificar comportamentos
    herdados. Nao carrega nenhum modelo real.
    """

    def carregar(self, dispositivo: str = "cpu") -> None:
        self._carregado = True

    def descarregar(self) -> None:
        self._carregado = False

    def detectar(self, imagem: Image.Image) -> ResultadoDeteccao:
        return ResultadoDeteccao(
            score=0.42,
            confianca=0.88,
            id_modelo=self.id_modelo,
            nome_modelo=self.nome_modelo,
        )


# ---------------------------------------------------------------------------
# Testes: ResultadoDeteccao
# ---------------------------------------------------------------------------


class TestResultadoDeteccao:
    """Testes de criacao e estrutura do dataclass ResultadoDeteccao."""

    def test_criacao_com_campos_obrigatorios(self) -> None:
        """Criacao com score, confianca, id_modelo e nome_modelo deve funcionar."""
        resultado = ResultadoDeteccao(
            score=0.75,
            confianca=0.92,
            id_modelo="modelo_teste",
            nome_modelo="Modelo de Teste",
        )

        assert resultado.score == 0.75
        assert resultado.confianca == 0.92
        assert resultado.id_modelo == "modelo_teste"
        assert resultado.nome_modelo == "Modelo de Teste"

    def test_campos_opcionais_com_valores_padrao(self) -> None:
        """Campos opcionais devem ter valores padrao corretos."""
        resultado = ResultadoDeteccao(
            score=0.50,
            confianca=0.80,
            id_modelo="teste",
            nome_modelo="Teste",
        )

        assert resultado.metadados == {}
        assert resultado.mapa_calor is None
        assert resultado.tempo_inferencia_ms == 0.0

    def test_criacao_com_todos_os_campos(self) -> None:
        """Criacao especificando todos os campos deve funcionar."""
        mapa_calor = np.zeros((224, 224), dtype=np.float32)
        resultado = ResultadoDeteccao(
            score=0.95,
            confianca=0.99,
            id_modelo="modelo_completo",
            nome_modelo="Modelo Completo",
            metadados={"versao": "1.0", "precisao": "fp16"},
            mapa_calor=mapa_calor,
            tempo_inferencia_ms=42.5,
        )

        assert resultado.score == 0.95
        assert resultado.metadados["versao"] == "1.0"
        assert resultado.mapa_calor is mapa_calor
        assert resultado.tempo_inferencia_ms == 42.5

    def test_metadados_independentes_entre_instancias(self) -> None:
        """
        O dict metadados deve ser independente entre instancias
        (nao deve ser um objeto compartilhado por default_factory).
        """
        resultado_a = ResultadoDeteccao(
            score=0.10, confianca=0.9, id_modelo="a", nome_modelo="A"
        )
        resultado_b = ResultadoDeteccao(
            score=0.90, confianca=0.9, id_modelo="b", nome_modelo="B"
        )

        resultado_a.metadados["chave"] = "valor_a"

        assert "chave" not in resultado_b.metadados


# ---------------------------------------------------------------------------
# Testes: DetectorBase
# ---------------------------------------------------------------------------


class TestDetectorBase:
    """Testes para a classe abstrata DetectorBase."""

    def test_detector_base_e_abstrato(self) -> None:
        """Nao deve ser possivel instanciar DetectorBase diretamente."""
        with pytest.raises(TypeError):
            DetectorBase("qualquer_id", "Qualquer Nome")  # type: ignore[abstract]

    def test_subclasse_concreta_pode_ser_instanciada(self) -> None:
        """Subclasse que implementa todos os metodos abstratos pode ser criada."""
        detector = DetectorConcreto("id_concreto", "Detector Concreto")

        assert detector.id_modelo == "id_concreto"
        assert detector.nome_modelo == "Detector Concreto"

    def test_propriedade_carregado_inicia_como_falso(self) -> None:
        """Propriedade carregado deve ser False antes de chamar carregar()."""
        detector = DetectorConcreto("teste", "Teste")

        assert detector.carregado is False

    def test_propriedade_carregado_verdadeiro_apos_carregar(self) -> None:
        """Propriedade carregado deve ser True apos chamar carregar()."""
        detector = DetectorConcreto("teste", "Teste")
        detector.carregar()

        assert detector.carregado is True

    def test_propriedade_carregado_falso_apos_descarregar(self) -> None:
        """Propriedade carregado deve voltar a False apos descarregar()."""
        detector = DetectorConcreto("teste", "Teste")
        detector.carregar()
        detector.descarregar()

        assert detector.carregado is False

    def test_treinar_levanta_not_implemented_error(self) -> None:
        """
        treinar() na implementacao padrao da classe base deve levantar
        NotImplementedError com mensagem informativa.
        """
        detector = DetectorConcreto("teste", "Teste")

        with pytest.raises(NotImplementedError) as exc_info:
            detector.treinar(dados_treino=None)

        assert "Fine-tuning" in str(exc_info.value) or "nao implementado" in str(
            exc_info.value
        ).lower()

    def test_repr_inclui_id_e_status(self) -> None:
        """__repr__ deve conter o id do modelo e o status de carregamento."""
        detector = DetectorConcreto("repr_teste", "Repr Teste")
        representacao = repr(detector)

        assert "repr_teste" in representacao
        assert "nao carregado" in representacao

    def test_repr_muda_apos_carregar(self) -> None:
        """__repr__ deve refletir o status atual do detector."""
        detector = DetectorConcreto("repr_teste_2", "Repr Teste 2")
        detector.carregar()
        representacao = repr(detector)

        assert "carregado" in representacao

    def test_detectar_retorna_resultado_deteccao(self) -> None:
        """detectar() deve retornar instancia de ResultadoDeteccao."""
        detector = DetectorConcreto("detectar_teste", "Detectar Teste")
        imagem = Image.fromarray(
            np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB"
        )

        resultado = detector.detectar(imagem)

        assert isinstance(resultado, ResultadoDeteccao)
        assert resultado.id_modelo == "detectar_teste"


# ---------------------------------------------------------------------------
# Testes: registro de modelos (REGISTRO_MODELOS)
# ---------------------------------------------------------------------------


class TestRegistroModelos:
    """Testes para o registro centralizado de modelos."""

    CAMPOS_OBRIGATORIOS_REGISTRO = {
        "id",
        "nome_exibicao",
        "hub_id",
        "arquitetura",
        "vram_fp16_mb",
        "papel",
        "tipo",
    }

    def test_registro_modelos_nao_vazio(self) -> None:
        """REGISTRO_MODELOS deve conter pelo menos um modelo."""
        assert len(REGISTRO_MODELOS) > 0

    def test_todos_modelos_tem_campos_obrigatorios(self) -> None:
        """Cada RegistroModelo deve possuir todos os campos obrigatorios."""
        for id_modelo, registro in REGISTRO_MODELOS.items():
            campos_presentes = {
                campo
                for campo in self.CAMPOS_OBRIGATORIOS_REGISTRO
                if hasattr(registro, campo)
            }
            ausentes = self.CAMPOS_OBRIGATORIOS_REGISTRO - campos_presentes

            assert not ausentes, (
                f"Modelo '{id_modelo}' esta sem os campos: {ausentes}"
            )

    def test_ids_consistentes_com_chave(self) -> None:
        """O campo id de cada RegistroModelo deve coincidir com sua chave no dicionario."""
        for chave, registro in REGISTRO_MODELOS.items():
            assert registro.id == chave, (
                f"Inconsistencia: chave='{chave}', registro.id='{registro.id}'"
            )

    def test_todos_tipos_sao_validos(self) -> None:
        """O campo tipo de cada modelo deve ser 'imagem', 'video' ou 'ambos'."""
        tipos_validos = {"imagem", "video", "ambos"}

        for id_modelo, registro in REGISTRO_MODELOS.items():
            assert registro.tipo in tipos_validos, (
                f"Modelo '{id_modelo}' tem tipo invalido: '{registro.tipo}'"
            )

    def test_vram_positiva_para_todos_os_modelos(self) -> None:
        """vram_fp16_mb deve ser positivo para todos os modelos registrados."""
        for id_modelo, registro in REGISTRO_MODELOS.items():
            assert registro.vram_fp16_mb > 0, (
                f"Modelo '{id_modelo}' tem vram_fp16_mb <= 0: {registro.vram_fp16_mb}"
            )

    def test_spatial_vit_registrado(self) -> None:
        """O modelo 'spatial_vit' deve estar presente no registro."""
        assert "spatial_vit" in REGISTRO_MODELOS

    def test_frequency_analyzer_registrado(self) -> None:
        """O modelo 'frequency_analyzer' deve estar presente no registro."""
        assert "frequency_analyzer" in REGISTRO_MODELOS


# ---------------------------------------------------------------------------
# Testes: obter_modelos_por_tipo
# ---------------------------------------------------------------------------


class TestObterModelosPorTipo:
    """Testes para a funcao de filtragem por tipo de midia."""

    def test_filtrar_por_imagem_retorna_somente_imagem(self) -> None:
        """obter_modelos_por_tipo('imagem') deve retornar apenas modelos de imagem."""
        modelos_imagem = obter_modelos_por_tipo("imagem")

        for id_modelo, registro in modelos_imagem.items():
            assert registro.tipo in ("imagem", "ambos"), (
                f"Modelo '{id_modelo}' nao e de imagem: tipo='{registro.tipo}'"
            )

    def test_filtrar_por_video_retorna_somente_video(self) -> None:
        """obter_modelos_por_tipo('video') deve retornar apenas modelos de video."""
        modelos_video = obter_modelos_por_tipo("video")

        for id_modelo, registro in modelos_video.items():
            assert registro.tipo in ("video", "ambos"), (
                f"Modelo '{id_modelo}' nao e de video: tipo='{registro.tipo}'"
            )

    def test_modelos_imagem_incluem_spatial_vit(self) -> None:
        """spatial_vit tem tipo='imagem' e deve aparecer na filtragem por imagem."""
        modelos_imagem = obter_modelos_por_tipo("imagem")

        assert "spatial_vit" in modelos_imagem

    def test_efficientnet_video_aparece_em_video(self) -> None:
        """efficientnet_video tem tipo='video' e deve aparecer na filtragem por video."""
        modelos_video = obter_modelos_por_tipo("video")

        assert "efficientnet_video" in modelos_video

    def test_modelos_desabilitados_sao_excluidos(self) -> None:
        """
        Modelos com habilitado=False nao devem aparecer nos resultados
        de obter_modelos_por_tipo.
        """
        # Verifica que a funcao nao inclui modelos com habilitado=False
        modelos = obter_modelos_por_tipo("imagem")

        for id_modelo, registro in modelos.items():
            assert registro.habilitado is True, (
                f"Modelo desabilitado '{id_modelo}' apareceu nos resultados"
            )

    def test_tipo_inexistente_retorna_dict_vazio(self) -> None:
        """Tipo que nao existe no registro deve retornar dicionario vazio."""
        modelos = obter_modelos_por_tipo("tipo_inexistente_xyz")

        assert modelos == {}


# ---------------------------------------------------------------------------
# Testes: calcular_vram_total
# ---------------------------------------------------------------------------


class TestCalcularVramTotal:
    """Testes para a funcao que calcula VRAM de pico necessaria."""

    def test_retorna_inteiro(self) -> None:
        """calcular_vram_total deve retornar um inteiro."""
        resultado = calcular_vram_total()

        assert isinstance(resultado, int)

    def test_retorna_valor_positivo(self) -> None:
        """VRAM total deve ser positiva."""
        resultado = calcular_vram_total()

        assert resultado > 0

    def test_igual_ao_maximo_das_vrams_individuais(self) -> None:
        """
        calcular_vram_total() calcula o pico sequencial (max), nao a soma.
        Deve ser igual ao max de vram_fp16_mb dos modelos habilitados.
        """
        vram_esperada = max(
            m.vram_fp16_mb
            for m in REGISTRO_MODELOS.values()
            if m.habilitado
        )
        resultado = calcular_vram_total()

        assert resultado == vram_esperada

    def test_nao_e_soma_das_vrams(self) -> None:
        """
        VRAM total deve ser o maximo, nao a soma (carregamento sequencial).
        """
        soma_vrams = sum(
            m.vram_fp16_mb
            for m in REGISTRO_MODELOS.values()
            if m.habilitado
        )
        resultado = calcular_vram_total()

        # Se ha mais de um modelo com VRAM > 0, o maximo deve ser < soma
        modelos_habilitados = [m for m in REGISTRO_MODELOS.values() if m.habilitado]
        if len(modelos_habilitados) > 1:
            assert resultado < soma_vrams, (
                f"calcular_vram_total ({resultado}) deveria ser menor que a "
                f"soma das vrams ({soma_vrams}) — carregamento e sequencial"
            )


# ---------------------------------------------------------------------------
# Testes: obter_modelo
# ---------------------------------------------------------------------------


class TestObterModelo:
    """Testes para a funcao de busca por ID."""

    def test_obter_modelo_existente(self) -> None:
        """ID existente deve retornar o RegistroModelo correto."""
        registro = obter_modelo("spatial_vit")

        assert registro is not None
        assert isinstance(registro, RegistroModelo)
        assert registro.id == "spatial_vit"

    def test_obter_modelo_inexistente_retorna_none(self) -> None:
        """ID que nao existe deve retornar None sem levantar excecao."""
        registro = obter_modelo("modelo_que_nao_existe_jamais")

        assert registro is None


# ---------------------------------------------------------------------------
# Testes: CalibradorConfianca
# ---------------------------------------------------------------------------


class TestCalibradorConfianca:
    """Testes para calibracao de scores e calculo de intervalos de confianca."""

    @pytest.fixture()
    def calibrador(self) -> CalibradorConfianca:
        """Calibrador com temperatura 1.0 (identidade para score 0.5)."""
        return CalibradorConfianca(temperatura=1.0)

    def test_calibrar_score_intermediario_permanece_aproximado(
        self, calibrador: CalibradorConfianca
    ) -> None:
        """
        Com temperatura=1.0, calibrar() e a composicao logit->sigmoid,
        que e a funcao identidade. Score 0.5 deve retornar 0.5.
        """
        score_calibrado = calibrador.calibrar(0.5)

        assert math.isclose(score_calibrado, 0.5, abs_tol=1e-6)

    def test_calibrar_retorna_float(
        self, calibrador: CalibradorConfianca
    ) -> None:
        """calibrar() deve retornar sempre um float."""
        resultado = calibrador.calibrar(0.7)

        assert isinstance(resultado, float)

    def test_calibrar_score_entre_zero_e_um(
        self, calibrador: CalibradorConfianca
    ) -> None:
        """Score calibrado deve sempre estar no intervalo [0, 1]."""
        for score_bruto in [0.0, 0.1, 0.5, 0.9, 1.0]:
            score_calibrado = calibrador.calibrar(score_bruto)
            assert 0.0 <= score_calibrado <= 1.0, (
                f"Score calibrado fora do intervalo para entrada {score_bruto}: "
                f"{score_calibrado}"
            )

    def test_temperatura_alta_suaviza_scores_extremos(self) -> None:
        """
        Temperatura > 1.0 aproxima os scores de 0.5 (suaviza predicoes).
        Score alto (0.9) com temperatura=3.0 deve ser menor do que com
        temperatura=1.0.
        """
        calibrador_suave = CalibradorConfianca(temperatura=3.0)
        calibrador_normal = CalibradorConfianca(temperatura=1.0)

        score_suave = calibrador_suave.calibrar(0.9)
        score_normal = calibrador_normal.calibrar(0.9)

        assert score_suave < score_normal, (
            f"Temperatura alta deveria suavizar score alto. "
            f"Temperatura=3.0: {score_suave:.4f}, temperatura=1.0: {score_normal:.4f}"
        )

    def test_temperatura_baixa_aguça_scores(self) -> None:
        """
        Temperatura < 1.0 torna predicoes mais extremas (scores altos
        ficam mais proximos de 1.0).
        """
        calibrador_agudo = CalibradorConfianca(temperatura=0.5)
        calibrador_normal = CalibradorConfianca(temperatura=1.0)

        score_agudo = calibrador_agudo.calibrar(0.7)
        score_normal = calibrador_normal.calibrar(0.7)

        assert score_agudo > score_normal, (
            f"Temperatura baixa deveria agucar score acima de 0.5. "
            f"Temperatura=0.5: {score_agudo:.4f}, temperatura=1.0: {score_normal:.4f}"
        )

    def test_intervalo_confianca_retorna_tupla_de_dois_floats(
        self, calibrador: CalibradorConfianca
    ) -> None:
        """calcular_intervalo_confianca deve retornar tupla (inferior, superior)."""
        intervalo = calibrador.calcular_intervalo_confianca(
            score=0.60, incerteza=0.10
        )

        assert isinstance(intervalo, tuple)
        assert len(intervalo) == 2

    def test_intervalo_inferior_menor_que_superior(
        self, calibrador: CalibradorConfianca
    ) -> None:
        """Limite inferior deve ser <= limite superior."""
        inferior, superior = calibrador.calcular_intervalo_confianca(
            score=0.60, incerteza=0.10
        )

        assert inferior <= superior

    def test_intervalo_dentro_de_zero_e_um(
        self, calibrador: CalibradorConfianca
    ) -> None:
        """Os limites do intervalo devem estar em [0, 1]."""
        inferior, superior = calibrador.calcular_intervalo_confianca(
            score=0.05, incerteza=0.50, nivel=0.99
        )

        assert 0.0 <= inferior <= 1.0
        assert 0.0 <= superior <= 1.0

    def test_incerteza_maior_produz_intervalo_mais_largo(
        self, calibrador: CalibradorConfianca
    ) -> None:
        """Score com maior incerteza deve ter intervalo de confianca mais amplo."""
        inf_baixa, sup_baixa = calibrador.calcular_intervalo_confianca(
            score=0.60, incerteza=0.05
        )
        inf_alta, sup_alta = calibrador.calcular_intervalo_confianca(
            score=0.60, incerteza=0.30
        )

        largura_incerteza_baixa = sup_baixa - inf_baixa
        largura_incerteza_alta = sup_alta - inf_alta

        assert largura_incerteza_alta >= largura_incerteza_baixa

    def test_calcular_concordancia_todos_iguais(
        self, calibrador: CalibradorConfianca
    ) -> None:
        """Modelos com scores na mesma faixa devem ter concordancia 1.0."""
        scores_concordantes = {
            "modelo_a": 0.80,  # faixa: ia
            "modelo_b": 0.85,  # faixa: ia
            "modelo_c": 0.88,  # faixa: ia
        }

        concordancia = calibrador.calcular_concordancia(scores_concordantes)

        assert math.isclose(concordancia, 1.0, abs_tol=1e-9)

    def test_calcular_concordancia_todos_discordam(
        self, calibrador: CalibradorConfianca
    ) -> None:
        """
        Modelos em faixas completamente diferentes devem ter concordancia < 1.0.
        """
        scores_discordantes = {
            "modelo_a": 0.10,  # faixa: real
            "modelo_b": 0.80,  # faixa: ia
        }

        concordancia = calibrador.calcular_concordancia(scores_discordantes)

        assert concordancia < 1.0

    def test_calcular_concordancia_modelo_unico(
        self, calibrador: CalibradorConfianca
    ) -> None:
        """Um unico modelo deve retornar concordancia 1.0."""
        concordancia = calibrador.calcular_concordancia({"modelo_x": 0.55})

        assert math.isclose(concordancia, 1.0, abs_tol=1e-9)

    def test_calcular_concordancia_dicionario_vazio(
        self, calibrador: CalibradorConfianca
    ) -> None:
        """Dicionario vazio deve retornar concordancia 1.0."""
        concordancia = calibrador.calcular_concordancia({})

        assert math.isclose(concordancia, 1.0, abs_tol=1e-9)
