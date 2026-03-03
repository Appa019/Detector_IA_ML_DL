"""
Agregacao de scores do ensemble de detectores.

Suporta dois modos:
1. Meta-learner (GradientBoosting) treinado sobre features dos detectores
2. Fallback: media ponderada com pesos configurados em settings.py
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from config.settings import CONFIG_ENSEMBLE, DIRETORIO_MODELOS
from models.base import ResultadoDeteccao

logger = logging.getLogger(__name__)

_CAMINHO_META_LEARNER = DIRETORIO_MODELOS / "ensemble" / "meta_learner.pkl"


@dataclass
class ResultadoEnsemble:
    """Resultado agregado do ensemble."""
    score_final: float  # 0.0 a 1.0
    classificacao: str  # "Provavelmente Real", etc.
    cor: str  # Cor hex para UI
    incerteza: float  # Desvio padrao entre modelos
    scores_individuais: Dict[str, float] = field(default_factory=dict)
    resultados_detalhados: List[ResultadoDeteccao] = field(default_factory=list)


class MetaAprendizEnsemble:
    """
    Meta-learner baseado em GradientBoosting para stacking do ensemble.

    Recebe scores e confiancas dos detectores individuais como features
    e prediz o score final calibrado. Quando nao treinado, retorna None
    e o AgregadorEnsemble usa media ponderada como fallback.

    Features do vetor de entrada (2N + 4):
    - score_1, confianca_1, ..., score_N, confianca_N (2N features)
    - std dos scores, max score, min score, range (4 features)
    """

    def __init__(self) -> None:
        self._modelo = None
        self._treinado = False
        self._carregar()

    def _carregar(self) -> None:
        """Carrega o meta-learner de disco se disponivel."""
        caminho = Path(_CAMINHO_META_LEARNER)
        if caminho.exists():
            try:
                import joblib
                self._modelo = joblib.load(caminho)
                self._treinado = True
                logger.info(f"Meta-learner carregado de: {caminho}")
            except Exception as excecao:
                logger.warning(
                    f"Falha ao carregar meta-learner ({caminho}): {excecao}. "
                    "Usando media ponderada como fallback."
                )
                self._treinado = False
        else:
            self._treinado = False
            logger.info(
                "Meta-learner nao encontrado. Usando media ponderada como fallback."
            )

    @property
    def treinado(self) -> bool:
        return self._treinado

    def predizer(self, resultados: List[ResultadoDeteccao]) -> Optional[float]:
        """
        Prediz score final usando o meta-learner treinado.

        Args:
            resultados: Lista de ResultadoDeteccao de cada modelo.

        Returns:
            Score final entre 0.0 e 1.0, ou None se nao treinado.
        """
        if not self._treinado or self._modelo is None:
            return None

        features = self._construir_features(resultados)
        try:
            proba = self._modelo.predict_proba(features.reshape(1, -1))
            # Indice 1 = classe IA/Fake
            score = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
            return max(0.0, min(1.0, score))
        except Exception as excecao:
            logger.error(f"Erro no meta-learner: {excecao}. Usando fallback.")
            return None

    def _construir_features(self, resultados: List[ResultadoDeteccao]) -> np.ndarray:
        """
        Constroi vetor de features para o meta-learner.

        Args:
            resultados: Lista de ResultadoDeteccao.

        Returns:
            Array numpy com features (2N + 4).
        """
        scores = []
        confiancas = []

        for resultado in resultados:
            scores.append(resultado.score)
            confiancas.append(resultado.confianca)

        scores_np = np.array(scores, dtype=np.float64)
        confiancas_np = np.array(confiancas, dtype=np.float64)

        # Intercala score, confianca para cada modelo
        pares = np.empty(len(scores) * 2, dtype=np.float64)
        pares[0::2] = scores_np
        pares[1::2] = confiancas_np

        # Estatisticas agregadas
        stats = np.array([
            float(scores_np.std()) if len(scores_np) > 1 else 0.0,
            float(scores_np.max()),
            float(scores_np.min()),
            float(scores_np.max() - scores_np.min()),
        ], dtype=np.float64)

        return np.concatenate([pares, stats])

    def treinar(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Treina o meta-learner com dados de calibracao.

        Args:
            X: Features (N_amostras, 2*N_modelos + 4).
            y: Rotulos binarios (0=real, 1=IA).

        Returns:
            Dicionario com metricas de treino.
        """
        from sklearn.ensemble import GradientBoostingClassifier
        import joblib

        self._modelo = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self._modelo.fit(X, y)
        self._treinado = True

        # Salva modelo
        caminho = Path(_CAMINHO_META_LEARNER)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._modelo, caminho)
        logger.info(f"Meta-learner salvo em: {caminho}")

        # Metricas
        score_treino = float(self._modelo.score(X, y))
        return {"acuracia_treino": score_treino}


class AgregadorEnsemble:
    """Agrega scores de multiplos detectores em um score final calibrado."""

    def __init__(self, pesos: Dict[str, float] = None):
        self.pesos = pesos or CONFIG_ENSEMBLE.pesos.copy()
        self._validar_pesos()
        self.meta_learner = MetaAprendizEnsemble()

    def _validar_pesos(self):
        """Valida que os pesos somam aproximadamente 1.0."""
        soma = sum(self.pesos.values())
        if abs(soma - 1.0) > 0.01:
            logger.warning(
                f"Pesos do ensemble somam {soma:.3f} (esperado: 1.0). Normalizando."
            )
            for chave in self.pesos:
                self.pesos[chave] /= soma

    def agregar(self, resultados: List[ResultadoDeteccao]) -> ResultadoEnsemble:
        """
        Agrega resultados de multiplos detectores.

        Usa meta-learner se treinado, senao media ponderada como fallback.

        Args:
            resultados: Lista de ResultadoDeteccao de cada modelo.

        Returns:
            ResultadoEnsemble com score final, classificacao e incerteza.
        """
        if not resultados:
            return ResultadoEnsemble(
                score_final=0.5,
                classificacao="Indeterminado",
                cor="#808080",
                incerteza=1.0,
            )

        # Coleta scores individuais
        scores = {}
        for resultado in resultados:
            scores[resultado.id_modelo] = resultado.score

        # Tenta meta-learner primeiro
        score_final_meta = self.meta_learner.predizer(resultados)

        if score_final_meta is not None:
            score_final = score_final_meta
            logger.debug(f"Ensemble: meta-learner score={score_final:.4f}")
        else:
            # Fallback: media ponderada
            score_final = self._media_ponderada(resultados)
            logger.debug(f"Ensemble: media ponderada score={score_final:.4f}")

        # Calcula incerteza (desvio padrao entre modelos)
        valores_scores = [r.score for r in resultados]
        incerteza = float(np.std(valores_scores)) if len(valores_scores) > 1 else 0.0

        # Classificacao
        classificacao, cor = self._classificar(score_final)

        return ResultadoEnsemble(
            score_final=float(score_final),
            classificacao=classificacao,
            cor=cor,
            incerteza=float(incerteza),
            scores_individuais=scores,
            resultados_detalhados=resultados,
        )

    def _media_ponderada(self, resultados: List[ResultadoDeteccao]) -> float:
        """Calcula media ponderada dos scores como fallback."""
        score_ponderado = 0.0
        peso_total = 0.0

        for resultado in resultados:
            peso = self.pesos.get(resultado.id_modelo, 0.0)
            if peso > 0:
                score_ponderado += peso * resultado.score
                peso_total += peso

        if peso_total > 0:
            return score_ponderado / peso_total
        return float(np.mean([r.score for r in resultados]))

    def _classificar(self, score: float) -> tuple:
        """Retorna classificacao e cor baseado no score."""
        thresholds = CONFIG_ENSEMBLE.thresholds

        if score < thresholds["provavelmente_real"]:
            return "Provavelmente Real", "#2ecc71"  # Verde
        elif score < thresholds["possivelmente_real"]:
            return "Possivelmente Real", "#82e0aa"  # Verde claro
        elif score < thresholds["possivelmente_ia"]:
            return "Possivelmente IA", "#f39c12"  # Laranja
        else:
            return "Provavelmente IA", "#e74c3c"  # Vermelho

    def agregar_temporal(
        self,
        resultados_frames: List[ResultadoEnsemble],
    ) -> ResultadoEnsemble:
        """
        Agrega resultados de multiplos frames de video.
        Usa media ponderada com peso maior para frames com maior confianca.
        """
        if not resultados_frames:
            return ResultadoEnsemble(
                score_final=0.5,
                classificacao="Indeterminado",
                cor="#808080",
                incerteza=1.0,
            )

        # Pondera frames pela inversa da incerteza (mais certos pesam mais)
        scores = np.array([r.score_final for r in resultados_frames])
        incertezas = np.array([max(r.incerteza, 0.01) for r in resultados_frames])
        pesos_frames = 1.0 / incertezas
        pesos_frames /= pesos_frames.sum()

        score_final = float(np.average(scores, weights=pesos_frames))
        incerteza_media = float(np.mean(incertezas))

        classificacao, cor = self._classificar(score_final)

        return ResultadoEnsemble(
            score_final=score_final,
            classificacao=classificacao,
            cor=cor,
            incerteza=incerteza_media,
        )
