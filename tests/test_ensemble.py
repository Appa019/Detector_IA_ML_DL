"""
Testes unitarios para o modulo de agregacao do ensemble.

Cobre AgregadorEnsemble, ResultadoEnsemble e a logica de classificacao
por faixas de score definidas em CONFIG_ENSEMBLE.thresholds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.ensemble import AgregadorEnsemble, ResultadoEnsemble
from models.base import ResultadoDeteccao


# ---------------------------------------------------------------------------
# Fixtures reutilizaveis
# ---------------------------------------------------------------------------


def _criar_resultado(
    id_modelo: str,
    score: float,
    confianca: float = 0.9,
) -> ResultadoDeteccao:
    """Fabrica auxiliar para ResultadoDeteccao com valores minimos."""
    return ResultadoDeteccao(
        score=score,
        confianca=confianca,
        id_modelo=id_modelo,
        nome_modelo=f"Modelo Teste {id_modelo}",
    )


@pytest.fixture()
def agregador_pesos_fixos() -> AgregadorEnsemble:
    """
    Agregador com pesos controlados para testes deterministicos.
    Os pesos somam exatamente 1.0, evitando normalizacao interna.
    """
    return AgregadorEnsemble(
        pesos={
            "modelo_a": 0.50,
            "modelo_b": 0.30,
            "modelo_c": 0.20,
        }
    )


# ---------------------------------------------------------------------------
# Testes: agregacao basica de scores
# ---------------------------------------------------------------------------


class TestAgregacaoBasica:
    """Testes de calculo do score ponderado."""

    def test_agregar_resultados_basico(
        self, agregador_pesos_fixos: AgregadorEnsemble
    ) -> None:
        """
        Tres detectores com scores e pesos conhecidos devem produzir
        a media ponderada exata.

        score_esperado = 0.50*0.80 + 0.30*0.60 + 0.20*0.40
                       = 0.40 + 0.18 + 0.08
                       = 0.66
        """
        resultados = [
            _criar_resultado("modelo_a", 0.80),
            _criar_resultado("modelo_b", 0.60),
            _criar_resultado("modelo_c", 0.40),
        ]

        resultado = agregador_pesos_fixos.agregar(resultados)

        assert isinstance(resultado, ResultadoEnsemble)
        assert math.isclose(resultado.score_final, 0.66, abs_tol=1e-9)

    def test_agregar_resultados_vazio(self) -> None:
        """Lista vazia deve retornar score 0.5 e classificacao Indeterminado."""
        agregador = AgregadorEnsemble()
        resultado = agregador.agregar([])

        assert resultado.score_final == 0.5
        assert resultado.classificacao == "Indeterminado"
        assert resultado.incerteza == 1.0

    def test_agregar_resultado_unico(self) -> None:
        """Um unico detector deve retornar seu proprio score como score_final."""
        agregador = AgregadorEnsemble(pesos={"modelo_x": 1.0})
        resultados = [_criar_resultado("modelo_x", 0.73)]

        resultado = agregador.agregar(resultados)

        assert math.isclose(resultado.score_final, 0.73, abs_tol=1e-9)

    def test_scores_individuais_preenchidos(
        self, agregador_pesos_fixos: AgregadorEnsemble
    ) -> None:
        """O dicionario scores_individuais deve mapear id_modelo -> score."""
        resultados = [
            _criar_resultado("modelo_a", 0.30),
            _criar_resultado("modelo_b", 0.70),
        ]
        resultado = agregador_pesos_fixos.agregar(resultados)

        assert resultado.scores_individuais["modelo_a"] == 0.30
        assert resultado.scores_individuais["modelo_b"] == 0.70

    def test_resultados_detalhados_preservados(
        self, agregador_pesos_fixos: AgregadorEnsemble
    ) -> None:
        """A lista resultados_detalhados deve conter os objetos originais."""
        r_a = _criar_resultado("modelo_a", 0.40)
        r_b = _criar_resultado("modelo_b", 0.60)

        resultado = agregador_pesos_fixos.agregar([r_a, r_b])

        assert r_a in resultado.resultados_detalhados
        assert r_b in resultado.resultados_detalhados

    def test_modelo_sem_peso_nao_influencia_score(self) -> None:
        """
        Detector cujo id nao esta em self.pesos recebe peso 0 e nao
        deve influenciar o score final ponderado.
        """
        agregador = AgregadorEnsemble(pesos={"modelo_registrado": 1.0})
        resultados = [
            _criar_resultado("modelo_registrado", 0.40),
            _criar_resultado("modelo_sem_peso", 0.99),  # nao deve pesar nada
        ]

        resultado = agregador.agregar(resultados)

        # Somente modelo_registrado tem peso; score final deve ser 0.40
        assert math.isclose(resultado.score_final, 0.40, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Testes: classificacao por faixas de score
# ---------------------------------------------------------------------------


class TestClassificacao:
    """Testes para as faixas de classificacao (thresholds do CONFIG_ENSEMBLE)."""

    # Thresholds: provavelmente_real=0.25, possivelmente_real=0.50,
    #             possivelmente_ia=0.75

    def test_classificacao_provavelmente_real(self) -> None:
        """Score < 0.25 deve ser classificado como 'Provavelmente Real'."""
        agregador = AgregadorEnsemble(pesos={"m": 1.0})
        resultado = agregador.agregar([_criar_resultado("m", 0.10)])

        assert resultado.classificacao == "Provavelmente Real"
        assert resultado.cor == "#2ecc71"

    def test_classificacao_possivelmente_real(self) -> None:
        """Score >= 0.25 e < 0.50 deve ser 'Possivelmente Real'."""
        agregador = AgregadorEnsemble(pesos={"m": 1.0})
        resultado = agregador.agregar([_criar_resultado("m", 0.37)])

        assert resultado.classificacao == "Possivelmente Real"
        assert resultado.cor == "#82e0aa"

    def test_classificacao_possivelmente_ia(self) -> None:
        """Score >= 0.50 e < 0.75 deve ser 'Possivelmente IA'."""
        agregador = AgregadorEnsemble(pesos={"m": 1.0})
        resultado = agregador.agregar([_criar_resultado("m", 0.62)])

        assert resultado.classificacao == "Possivelmente IA"
        assert resultado.cor == "#f39c12"

    def test_classificacao_provavelmente_ia(self) -> None:
        """Score >= 0.75 deve ser classificado como 'Provavelmente IA'."""
        agregador = AgregadorEnsemble(pesos={"m": 1.0})
        resultado = agregador.agregar([_criar_resultado("m", 0.90)])

        assert resultado.classificacao == "Provavelmente IA"
        assert resultado.cor == "#e74c3c"

    def test_limiar_provavelmente_real_exato(self) -> None:
        """Score exatamente 0.25 deve estar na faixa 'Possivelmente Real'."""
        agregador = AgregadorEnsemble(pesos={"m": 1.0})
        resultado = agregador.agregar([_criar_resultado("m", 0.25)])

        assert resultado.classificacao == "Possivelmente Real"

    def test_limiar_possivelmente_ia_exato(self) -> None:
        """Score exatamente 0.75 deve estar na faixa 'Provavelmente IA'."""
        agregador = AgregadorEnsemble(pesos={"m": 1.0})
        resultado = agregador.agregar([_criar_resultado("m", 0.75)])

        assert resultado.classificacao == "Provavelmente IA"


# ---------------------------------------------------------------------------
# Testes: incerteza (desvio padrao entre modelos)
# ---------------------------------------------------------------------------


class TestIncerteza:
    """Testes para o calculo de incerteza do ensemble."""

    def test_incerteza_alta_quando_modelos_discordam(
        self, agregador_pesos_fixos: AgregadorEnsemble
    ) -> None:
        """
        Modelos com scores muito diferentes devem gerar alta incerteza.
        Desvio padrao de [0.05, 0.95] e aproximadamente 0.45.
        """
        resultados = [
            _criar_resultado("modelo_a", 0.05),
            _criar_resultado("modelo_b", 0.95),
            _criar_resultado("modelo_c", 0.50),
        ]
        resultado = agregador_pesos_fixos.agregar(resultados)

        assert resultado.incerteza > 0.30, (
            f"Incerteza esperada > 0.30 para modelos discordantes, "
            f"obtida: {resultado.incerteza:.4f}"
        )

    def test_incerteza_baixa_quando_modelos_concordam(
        self, agregador_pesos_fixos: AgregadorEnsemble
    ) -> None:
        """
        Modelos com scores muito proximos devem gerar baixa incerteza.
        Desvio padrao de [0.79, 0.80, 0.81] e aprox. 0.008.
        """
        resultados = [
            _criar_resultado("modelo_a", 0.79),
            _criar_resultado("modelo_b", 0.80),
            _criar_resultado("modelo_c", 0.81),
        ]
        resultado = agregador_pesos_fixos.agregar(resultados)

        assert resultado.incerteza < 0.05, (
            f"Incerteza esperada < 0.05 para modelos concordantes, "
            f"obtida: {resultado.incerteza:.4f}"
        )

    def test_incerteza_zero_para_resultado_unico(self) -> None:
        """Um unico resultado tem incerteza 0 (desvio padrao de um elemento)."""
        agregador = AgregadorEnsemble(pesos={"modelo_x": 1.0})
        resultado = agregador.agregar([_criar_resultado("modelo_x", 0.60)])

        assert resultado.incerteza == 0.0

    def test_incerteza_coerente_com_desvio_padrao(
        self, agregador_pesos_fixos: AgregadorEnsemble
    ) -> None:
        """
        A incerteza deve ser igual ao desvio padrao (populacional)
        dos scores brutos dos detectores.
        """
        scores_brutos = [0.20, 0.50, 0.80]
        resultados = [
            _criar_resultado("modelo_a", scores_brutos[0]),
            _criar_resultado("modelo_b", scores_brutos[1]),
            _criar_resultado("modelo_c", scores_brutos[2]),
        ]
        resultado = agregador_pesos_fixos.agregar(resultados)

        desvio_esperado = float(np.std(scores_brutos))
        assert math.isclose(resultado.incerteza, desvio_esperado, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Testes: agregacao temporal (frames de video)
# ---------------------------------------------------------------------------


class TestAgregacaoTemporal:
    """Testes para agregacao de resultados de multiplos frames de video."""

    @pytest.fixture()
    def agregador(self) -> AgregadorEnsemble:
        return AgregadorEnsemble()

    def _criar_resultado_ensemble(
        self, score: float, incerteza: float
    ) -> ResultadoEnsemble:
        """Cria ResultadoEnsemble sintetico para testes temporais."""
        agregador = AgregadorEnsemble(pesos={"m": 1.0})
        # Criamos diretamente para ter controle sobre incerteza
        return ResultadoEnsemble(
            score_final=score,
            classificacao="Teste",
            cor="#000000",
            incerteza=incerteza,
        )

    def test_agregar_temporal_basico(self, agregador: AgregadorEnsemble) -> None:
        """Agregacao temporal de frames deve retornar ResultadoEnsemble valido."""
        frames = [
            self._criar_resultado_ensemble(0.70, 0.10),
            self._criar_resultado_ensemble(0.80, 0.10),
            self._criar_resultado_ensemble(0.75, 0.10),
        ]
        resultado = agregador.agregar_temporal(frames)

        assert isinstance(resultado, ResultadoEnsemble)
        assert 0.0 <= resultado.score_final <= 1.0
        assert resultado.classificacao != ""

    def test_agregar_temporal_vazio(self, agregador: AgregadorEnsemble) -> None:
        """Lista vazia de frames deve retornar score 0.5 e Indeterminado."""
        resultado = agregador.agregar_temporal([])

        assert resultado.score_final == 0.5
        assert resultado.classificacao == "Indeterminado"

    def test_agregar_temporal_frame_unico(self, agregador: AgregadorEnsemble) -> None:
        """Frame unico deve produzir score_final igual ao score do frame."""
        frames = [self._criar_resultado_ensemble(0.85, 0.05)]
        resultado = agregador.agregar_temporal(frames)

        assert math.isclose(resultado.score_final, 0.85, abs_tol=1e-9)

    def test_agregar_temporal_frame_certeza_alta_pesa_mais(
        self, agregador: AgregadorEnsemble
    ) -> None:
        """
        Frame com baixa incerteza (alta certeza) deve pesar mais
        na media ponderada. Score final deve ser mais proximo do
        frame com incerteza menor.

        frame_certo:   score=0.90, incerteza=0.01 -> peso alto
        frame_incerto: score=0.10, incerteza=0.50 -> peso baixo
        Score final deve ser proximo de 0.90.
        """
        frames = [
            self._criar_resultado_ensemble(0.90, 0.01),  # frame mais certo
            self._criar_resultado_ensemble(0.10, 0.50),  # frame muito incerto
        ]
        resultado = agregador.agregar_temporal(frames)

        assert resultado.score_final > 0.70, (
            f"Frame com alta certeza deveria dominar o score. "
            f"Score obtido: {resultado.score_final:.4f}"
        )

    def test_agregar_temporal_score_dentro_intervalo(
        self, agregador: AgregadorEnsemble
    ) -> None:
        """Score final temporal deve estar sempre entre 0.0 e 1.0."""
        scores_extremos = [0.0, 0.5, 1.0]
        frames = [
            self._criar_resultado_ensemble(s, 0.10) for s in scores_extremos
        ]
        resultado = agregador.agregar_temporal(frames)

        assert 0.0 <= resultado.score_final <= 1.0
