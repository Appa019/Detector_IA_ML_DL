"""
Calibracao e calculo de confianca dos scores do ensemble.

Suporta dois modos de calibracao:
1. Calibracao aprendida (IsotonicRegression) — quando treinado em dados reais
2. Fallback: Temperature Scaling — calibracao parametrica simples
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from config.settings import CONFIG_ENSEMBLE, DIRETORIO_MODELOS

logger = logging.getLogger(__name__)

_CAMINHO_CALIBRADOR = DIRETORIO_MODELOS / "calibracao" / "calibrador.pkl"


class CalibradorConfianca:
    """
    Calibra scores brutos usando IsotonicRegression (se treinado)
    ou Temperature Scaling (fallback).
    """

    def __init__(self, temperatura: float = None):
        self.temperatura = temperatura or CONFIG_ENSEMBLE.temperatura
        self._calibrador_isotonico = None
        self._isotonico_treinado = False
        self._carregar_calibrador()

    def _carregar_calibrador(self) -> None:
        """Carrega calibrador isotonico de disco se disponivel."""
        caminho = Path(_CAMINHO_CALIBRADOR)
        if caminho.exists():
            try:
                import joblib
                self._calibrador_isotonico = joblib.load(caminho)
                self._isotonico_treinado = True
                logger.info(f"Calibrador isotonico carregado de: {caminho}")
            except Exception as excecao:
                logger.warning(
                    f"Falha ao carregar calibrador ({caminho}): {excecao}. "
                    "Usando Temperature Scaling como fallback."
                )
                self._isotonico_treinado = False
        else:
            self._isotonico_treinado = False
            logger.info(
                "Calibrador isotonico nao encontrado. Usando Temperature Scaling."
            )

    def calibrar(self, score_bruto: float) -> float:
        """
        Calibra o score usando isotonico (se treinado) ou Temperature Scaling.

        Args:
            score_bruto: Score entre 0.0 e 1.0.

        Returns:
            Score calibrado entre 0.0 e 1.0.
        """
        if self._isotonico_treinado and self._calibrador_isotonico is not None:
            return self._calibrar_isotonico(score_bruto)
        return self._calibrar_temperatura(score_bruto)

    def _calibrar_isotonico(self, score_bruto: float) -> float:
        """Calibra usando IsotonicRegression treinada."""
        try:
            score_np = np.array([score_bruto])
            score_calibrado = float(self._calibrador_isotonico.predict(score_np)[0])
            return max(0.0, min(1.0, score_calibrado))
        except Exception as excecao:
            logger.warning(
                f"Erro na calibracao isotonica: {excecao}. "
                "Fallback para Temperature Scaling."
            )
            return self._calibrar_temperatura(score_bruto)

    def _calibrar_temperatura(self, score_bruto: float) -> float:
        """
        Aplica Temperature Scaling para calibrar o score.

        Args:
            score_bruto: Score entre 0.0 e 1.0.

        Returns:
            Score calibrado entre 0.0 e 1.0.
        """
        # Converte para logit (inversa da sigmoid)
        score_clamp = np.clip(score_bruto, 1e-7, 1.0 - 1e-7)
        logit = np.log(score_clamp / (1.0 - score_clamp))

        # Aplica temperatura
        logit_calibrado = logit / self.temperatura

        # Converte de volta para probabilidade
        score_calibrado = 1.0 / (1.0 + np.exp(-logit_calibrado))

        return float(score_calibrado)

    def treinar_calibracao(
        self,
        scores_brutos: np.ndarray,
        rotulos: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Treina calibracao isotonica com dados de validacao.

        Args:
            scores_brutos: Array de scores brutos (N,).
            rotulos: Array de rotulos binarios 0/1 (N,).

        Returns:
            Dicionario com metricas de calibracao.
        """
        from sklearn.isotonic import IsotonicRegression
        import joblib

        self._calibrador_isotonico = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )
        self._calibrador_isotonico.fit(scores_brutos, rotulos)
        self._isotonico_treinado = True

        # Salva calibrador
        caminho = Path(_CAMINHO_CALIBRADOR)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._calibrador_isotonico, caminho)
        logger.info(f"Calibrador isotonico salvo em: {caminho}")

        # Metricas
        scores_calibrados = self._calibrador_isotonico.predict(scores_brutos)
        erro_calibracao = float(np.mean(np.abs(scores_calibrados - rotulos)))

        return {
            "metodo": "IsotonicRegression",
            "erro_calibracao_medio": erro_calibracao,
            "n_amostras": len(rotulos),
        }

    def calcular_intervalo_confianca(
        self,
        score: float,
        incerteza: float,
        nivel: float = 0.95,
    ) -> tuple:
        """
        Calcula intervalo de confianca baseado na incerteza do ensemble.

        Args:
            score: Score final do ensemble.
            incerteza: Desvio padrao entre modelos.
            nivel: Nivel de confianca (padrao 95%).

        Returns:
            Tupla (limite_inferior, limite_superior).
        """
        # Z-score para o nivel de confianca
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(nivel, 1.96)

        margem = z * incerteza
        limite_inferior = max(0.0, score - margem)
        limite_superior = min(1.0, score + margem)

        return (float(limite_inferior), float(limite_superior))

    def calcular_concordancia(self, scores: dict) -> float:
        """
        Calcula indice de concordancia entre os modelos.
        1.0 = todos concordam, 0.0 = total discordancia.
        """
        if len(scores) <= 1:
            return 1.0

        valores = list(scores.values())
        # Todos na mesma faixa de classificacao?
        classificacoes = [self._faixa(s) for s in valores]
        concordantes = classificacoes.count(classificacoes[0])
        return concordantes / len(classificacoes)

    def _faixa(self, score: float) -> str:
        """Retorna a faixa de classificacao de um score."""
        thresholds = CONFIG_ENSEMBLE.thresholds
        if score < thresholds["provavelmente_real"]:
            return "real"
        elif score < thresholds["possivelmente_real"]:
            return "possivelmente_real"
        elif score < thresholds["possivelmente_ia"]:
            return "possivelmente_ia"
        else:
            return "ia"
