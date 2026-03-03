"""
Testes unitarios para o modulo de analise espectral de imagens.

Todos os testes usam imagens sinteticas geradas com numpy — sem acesso a
rede, GPU ou modelos pre-treinados. Cada teste deve executar em < 1 segundo.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.frequency import AnalisadorEspectral

# Semente fixa para resultados reprodutiveis nos testes
SEMENTE_ALEATORIA = 42


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def analisador() -> AnalisadorEspectral:
    """Instancia limpa do AnalisadorEspectral para cada teste."""
    return AnalisadorEspectral()


@pytest.fixture()
def imagem_ruidosa_256() -> np.ndarray:
    """Imagem RGB 256x256 com ruido branco uniforme."""
    rng = np.random.default_rng(SEMENTE_ALEATORIA)
    return (rng.random((256, 256, 3)) * 255).astype(np.uint8)


@pytest.fixture()
def imagem_uniforme_256() -> np.ndarray:
    """
    Imagem RGB 256x256 com valor constante (128 em todos os pixels).
    Representa o caso extremo de imagem 'lisa' sem variacao de textura.
    """
    return np.full((256, 256, 3), 128, dtype=np.uint8)


@pytest.fixture()
def imagem_cinza_256() -> np.ndarray:
    """Imagem em escala de cinza 256x256 com valores aleatorios."""
    rng = np.random.default_rng(SEMENTE_ALEATORIA)
    return (rng.random((256, 256)) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Testes: calcular_fft_2d
# ---------------------------------------------------------------------------


class TestCalcularFFT2D:
    """Testes para o calculo do espectro de magnitude via FFT."""

    def test_shape_saida_igual_ao_shape_entrada_rgb(
        self, analisador: AnalisadorEspectral, imagem_ruidosa_256: np.ndarray
    ) -> None:
        """FFT de imagem RGB (H, W, 3) deve retornar espectro 2D (H, W)."""
        espectro = analisador.calcular_fft_2d(imagem_ruidosa_256)

        assert espectro.ndim == 2
        assert espectro.shape == (256, 256)

    def test_shape_saida_igual_ao_shape_entrada_cinza(
        self, analisador: AnalisadorEspectral, imagem_cinza_256: np.ndarray
    ) -> None:
        """FFT de imagem em escala de cinza (H, W) deve retornar espectro (H, W)."""
        espectro = analisador.calcular_fft_2d(imagem_cinza_256)

        assert espectro.ndim == 2
        assert espectro.shape == (256, 256)

    def test_espectro_valores_nao_negativos(
        self, analisador: AnalisadorEspectral, imagem_ruidosa_256: np.ndarray
    ) -> None:
        """
        Espectro em escala log (log1p(magnitude)) nao pode conter
        valores negativos, pois log1p(x >= 0) >= 0.
        """
        espectro = analisador.calcular_fft_2d(imagem_ruidosa_256)

        assert np.all(espectro >= 0), "Espectro FFT continha valores negativos"

    def test_imagem_uniforme_espectro_concentrado_no_centro(
        self, analisador: AnalisadorEspectral, imagem_uniforme_256: np.ndarray
    ) -> None:
        """
        Imagem uniforme tem toda a energia na componente DC (centro do
        espectro centralizado). As demais posicoes devem ser zero.
        """
        espectro = analisador.calcular_fft_2d(imagem_uniforme_256)
        centro_y, centro_x = 128, 128

        # Centro deve ter valor alto
        assert espectro[centro_y, centro_x] > 0

        # Media das bordas deve ser (praticamente) zero
        borda_superior = espectro[0, :].mean()
        assert borda_superior < 1e-6, (
            f"Bordas deveriam ser proximas de zero para imagem uniforme, "
            f"obtido: {borda_superior:.6f}"
        )

    def test_imagens_diferentes_produzem_espectros_diferentes(
        self,
        analisador: AnalisadorEspectral,
        imagem_ruidosa_256: np.ndarray,
        imagem_uniforme_256: np.ndarray,
    ) -> None:
        """Imagens com conteudo diferente devem ter espectros diferentes."""
        espectro_ruidoso = analisador.calcular_fft_2d(imagem_ruidosa_256)
        espectro_uniforme = analisador.calcular_fft_2d(imagem_uniforme_256)

        assert not np.allclose(espectro_ruidoso, espectro_uniforme)

    def test_imagem_tamanho_arbitrario(self, analisador: AnalisadorEspectral) -> None:
        """FFT deve funcionar para imagens de tamanhos nao-potencia-de-dois."""
        rng = np.random.default_rng(SEMENTE_ALEATORIA)
        imagem = rng.integers(0, 256, (100, 150, 3), dtype=np.uint8)

        espectro = analisador.calcular_fft_2d(imagem)

        assert espectro.shape == (100, 150)


# ---------------------------------------------------------------------------
# Testes: calcular_dct_2d
# ---------------------------------------------------------------------------


class TestCalcularDCT2D:
    """Testes para o calculo da Transformada Discreta do Cosseno 2D."""

    def test_shape_saida_igual_ao_shape_entrada_rgb(
        self, analisador: AnalisadorEspectral, imagem_ruidosa_256: np.ndarray
    ) -> None:
        """DCT de imagem RGB deve retornar coeficientes com shape (H, W)."""
        coeficientes = analisador.calcular_dct_2d(imagem_ruidosa_256)

        assert coeficientes.ndim == 2
        assert coeficientes.shape == (256, 256)

    def test_coeficientes_sao_float(
        self, analisador: AnalisadorEspectral, imagem_ruidosa_256: np.ndarray
    ) -> None:
        """Coeficientes DCT devem ser valores de ponto flutuante."""
        coeficientes = analisador.calcular_dct_2d(imagem_ruidosa_256)

        assert np.issubdtype(coeficientes.dtype, np.floating)

    def test_imagem_uniforme_coeficiente_dc_dominante(
        self, analisador: AnalisadorEspectral, imagem_uniforme_256: np.ndarray
    ) -> None:
        """
        Para imagem uniforme, o coeficiente DC (posicao [0, 0]) deve ser
        muito maior em modulo do que os coeficientes de AC.
        """
        coeficientes = analisador.calcular_dct_2d(imagem_uniforme_256)

        coef_dc = abs(coeficientes[0, 0])
        media_ac = np.abs(coeficientes[1:, 1:]).mean()

        assert coef_dc > media_ac * 10, (
            f"Coeficiente DC ({coef_dc:.2f}) nao domina os coeficientes AC "
            f"(media AC: {media_ac:.4f}) para imagem uniforme"
        )

    def test_dct_imagens_diferentes_produzem_coeficientes_diferentes(
        self,
        analisador: AnalisadorEspectral,
        imagem_ruidosa_256: np.ndarray,
        imagem_uniforme_256: np.ndarray,
    ) -> None:
        """Imagens com conteudos diferentes devem ter DCTs diferentes."""
        dct_ruidosa = analisador.calcular_dct_2d(imagem_ruidosa_256)
        dct_uniforme = analisador.calcular_dct_2d(imagem_uniforme_256)

        assert not np.allclose(dct_ruidosa, dct_uniforme)


# ---------------------------------------------------------------------------
# Testes: calcular_media_azimuthal
# ---------------------------------------------------------------------------


class TestCalcularMediaAzimuthal:
    """Testes para o perfil azimutal medio do espectro."""

    def test_saida_e_array_1d(
        self, analisador: AnalisadorEspectral, imagem_ruidosa_256: np.ndarray
    ) -> None:
        """Perfil azimutal deve ser um array 1D."""
        espectro = analisador.calcular_fft_2d(imagem_ruidosa_256)
        perfil = analisador.calcular_media_azimuthal(espectro)

        assert perfil.ndim == 1

    def test_comprimento_igual_ao_raio_maximo(
        self, analisador: AnalisadorEspectral
    ) -> None:
        """
        O comprimento do perfil azimutal deve ser igual ao raio maximo,
        definido como min(altura // 2, largura // 2) para espectro centralizado.
        """
        # Espectro quadrado 64x64: raio_maximo = 32
        espectro_quadrado = np.ones((64, 64))
        perfil = analisador.calcular_media_azimuthal(espectro_quadrado)

        assert len(perfil) == 32

    def test_comprimento_para_espectro_retangular(
        self, analisador: AnalisadorEspectral
    ) -> None:
        """Para espectro retangular, comprimento deve ser min das meias-dimensoes."""
        # Espectro 100x200: centro_y=50, centro_x=100; raio_max = min(50,100) = 50
        espectro_retangular = np.ones((100, 200))
        perfil = analisador.calcular_media_azimuthal(espectro_retangular)

        assert len(perfil) == 50

    def test_perfil_valores_nao_negativos_para_espectro_positivo(
        self, analisador: AnalisadorEspectral, imagem_ruidosa_256: np.ndarray
    ) -> None:
        """
        Se o espectro de entrada e nao-negativo (como saida de calcular_fft_2d),
        o perfil azimutal tambem deve ser nao-negativo.
        """
        espectro = analisador.calcular_fft_2d(imagem_ruidosa_256)
        perfil = analisador.calcular_media_azimuthal(espectro)

        assert np.all(perfil >= 0)

    def test_perfil_espectro_uniforme_constante(
        self, analisador: AnalisadorEspectral
    ) -> None:
        """
        Espectro com valor constante deve produzir perfil azimutal constante
        (todas as medias de anel iguais ao valor constante).
        """
        valor_constante = 5.0
        espectro = np.full((64, 64), valor_constante)
        perfil = analisador.calcular_media_azimuthal(espectro)

        assert np.allclose(perfil, valor_constante, atol=1e-10)


# ---------------------------------------------------------------------------
# Testes: extrair_features_frequencia
# ---------------------------------------------------------------------------


class TestExtrairFeaturesFrequencia:
    """Testes para o dicionario de features espectrais."""

    CHAVES_ESPERADAS = {
        "media_espectro",
        "desvio_espectro",
        "assimetria_espectro",
        "curtose_espectro",
        "razao_hf_lf",
        "inclinacao_espectral",
        "media_perfil_alta_freq",
        "media_perfil_baixa_freq",
        "energia_lf_dct",
        "energia_total_dct",
        "razao_energia_dct",
    }

    def test_retorna_dicionario_com_todas_as_chaves(
        self, analisador: AnalisadorEspectral, imagem_ruidosa_256: np.ndarray
    ) -> None:
        """extrair_features_frequencia deve retornar dict com todas as chaves."""
        features = analisador.extrair_features_frequencia(imagem_ruidosa_256)

        assert isinstance(features, dict)
        assert self.CHAVES_ESPERADAS.issubset(features.keys()), (
            f"Chaves ausentes: {self.CHAVES_ESPERADAS - features.keys()}"
        )

    def test_todos_valores_sao_floats(
        self, analisador: AnalisadorEspectral, imagem_ruidosa_256: np.ndarray
    ) -> None:
        """Todos os valores do dicionario de features devem ser float."""
        features = analisador.extrair_features_frequencia(imagem_ruidosa_256)

        for chave, valor in features.items():
            assert isinstance(valor, float), (
                f"Feature '{chave}' deveria ser float, obtido: {type(valor)}"
            )

    def test_features_com_imagem_uniforme(
        self, analisador: AnalisadorEspectral, imagem_uniforme_256: np.ndarray
    ) -> None:
        """
        Imagem uniforme nao deve levantar excecao; razao_hf_lf deve ser
        proxima de zero pois toda energia esta na componente DC.
        """
        features = analisador.extrair_features_frequencia(imagem_uniforme_256)

        assert isinstance(features, dict)
        assert self.CHAVES_ESPERADAS.issubset(features.keys())
        # Imagem uniforme: quase toda energia em DC -> razao_hf_lf ~ 0
        assert features["razao_hf_lf"] < 0.10, (
            f"razao_hf_lf para imagem uniforme deveria ser pequena, "
            f"obtida: {features['razao_hf_lf']:.4f}"
        )

    def test_desvio_espectro_positivo_para_imagem_nao_uniforme(
        self, analisador: AnalisadorEspectral, imagem_ruidosa_256: np.ndarray
    ) -> None:
        """Imagem com variacao deve ter desvio espectral positivo."""
        features = analisador.extrair_features_frequencia(imagem_ruidosa_256)

        assert features["desvio_espectro"] > 0

    def test_energia_total_dct_nao_negativa(
        self, analisador: AnalisadorEspectral, imagem_ruidosa_256: np.ndarray
    ) -> None:
        """Energia total DCT e media de valores absolutos, logo nao-negativa."""
        features = analisador.extrair_features_frequencia(imagem_ruidosa_256)

        assert features["energia_total_dct"] >= 0


# ---------------------------------------------------------------------------
# Testes: calcular_razao_frequencia
# ---------------------------------------------------------------------------


class TestCalcularRazaoFrequencia:
    """Testes para a razao de energia alta/baixa frequencia."""

    def test_retorna_float(
        self, analisador: AnalisadorEspectral, imagem_ruidosa_256: np.ndarray
    ) -> None:
        """calcular_razao_frequencia deve retornar um float."""
        espectro = analisador.calcular_fft_2d(imagem_ruidosa_256)
        razao = analisador.calcular_razao_frequencia(espectro)

        assert isinstance(razao, float)

    def test_razao_entre_zero_e_um(
        self, analisador: AnalisadorEspectral, imagem_ruidosa_256: np.ndarray
    ) -> None:
        """
        A razao energia_hf / energia_total deve estar em [0, 1].
        Como espectro FFT e nao-negativo, a razao e valida neste intervalo.
        """
        espectro = analisador.calcular_fft_2d(imagem_ruidosa_256)
        razao = analisador.calcular_razao_frequencia(espectro)

        assert 0.0 <= razao <= 1.0, (
            f"Razao de frequencia fora do intervalo [0, 1]: {razao:.6f}"
        )

    def test_limiar_zero_retorna_razao_um(
        self, analisador: AnalisadorEspectral, imagem_ruidosa_256: np.ndarray
    ) -> None:
        """
        Com limiar=0.0, toda a energia e considerada alta frequencia
        (mascara_hf cobre todo o espectro), portanto razao ~ 1.0.
        """
        espectro = analisador.calcular_fft_2d(imagem_ruidosa_256)
        razao = analisador.calcular_razao_frequencia(espectro, limiar=0.0)

        # Limiar zero: distancia > 0 inclui quase tudo (exceto o centro exato)
        assert razao > 0.95, (
            f"Com limiar=0, razao esperada proxima de 1.0, obtida: {razao:.4f}"
        )

    def test_espectro_3d_levanta_value_error(
        self, analisador: AnalisadorEspectral
    ) -> None:
        """Espectro com ndim != 2 deve levantar ValueError."""
        espectro_invalido = np.ones((64, 64, 3))

        with pytest.raises(ValueError, match="2D"):
            analisador.calcular_razao_frequencia(espectro_invalido)

    def test_razao_consistente_entre_chamadas(
        self, analisador: AnalisadorEspectral, imagem_ruidosa_256: np.ndarray
    ) -> None:
        """O mesmo espectro deve produzir a mesma razao em chamadas sucessivas."""
        espectro = analisador.calcular_fft_2d(imagem_ruidosa_256)

        razao_1 = analisador.calcular_razao_frequencia(espectro)
        razao_2 = analisador.calcular_razao_frequencia(espectro)

        assert razao_1 == razao_2


# ---------------------------------------------------------------------------
# Testes comparativos: imagem uniforme vs. imagem ruidosa
# ---------------------------------------------------------------------------


class TestComparativoUniformeVsRuidosa:
    """
    Testes que verificam as diferencas esperadas entre imagens
    artificialmente lisas (baixa alta-frequencia) e ruidosas.
    """

    def test_imagem_uniforme_tem_razao_hf_lf_menor(
        self,
        analisador: AnalisadorEspectral,
        imagem_uniforme_256: np.ndarray,
        imagem_ruidosa_256: np.ndarray,
    ) -> None:
        """
        Imagem uniforme deve ter razao_hf_lf significativamente menor
        do que imagem com ruido branco — base da heuristica de deteccao de IA.
        """
        espectro_uniforme = analisador.calcular_fft_2d(imagem_uniforme_256)
        espectro_ruidoso = analisador.calcular_fft_2d(imagem_ruidosa_256)

        razao_uniforme = analisador.calcular_razao_frequencia(espectro_uniforme)
        razao_ruidosa = analisador.calcular_razao_frequencia(espectro_ruidoso)

        assert razao_uniforme < razao_ruidosa, (
            f"Imagem uniforme (razao={razao_uniforme:.4f}) deveria ter menor "
            f"razao HF/LF do que imagem ruidosa (razao={razao_ruidosa:.4f})"
        )

    def test_imagem_ruidosa_tem_features_desvio_maior(
        self,
        analisador: AnalisadorEspectral,
        imagem_uniforme_256: np.ndarray,
        imagem_ruidosa_256: np.ndarray,
    ) -> None:
        """
        Imagem ruidosa deve ter maior desvio_espectro do que imagem uniforme.
        """
        features_uniforme = analisador.extrair_features_frequencia(imagem_uniforme_256)
        features_ruidosa = analisador.extrair_features_frequencia(imagem_ruidosa_256)

        assert features_ruidosa["desvio_espectro"] > features_uniforme["desvio_espectro"], (
            "Imagem ruidosa deveria ter maior desvio espectral do que imagem uniforme"
        )

    def test_imagem_gradiente_entre_uniforme_e_ruidosa(
        self, analisador: AnalisadorEspectral
    ) -> None:
        """
        Imagem com gradiente suave (sinal de baixa frequencia) deve ter
        razao_hf_lf menor do que imagem totalmente aleatoria.
        """
        rng = np.random.default_rng(SEMENTE_ALEATORIA)

        # Gradiente horizontal suave em RGB
        gradiente = np.tile(
            np.linspace(0, 255, 256, dtype=np.uint8), (256, 1)
        )
        imagem_gradiente = np.stack([gradiente, gradiente, gradiente], axis=-1)

        # Ruido branco puro
        imagem_ruido_branco = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)

        espectro_gradiente = analisador.calcular_fft_2d(imagem_gradiente)
        espectro_ruido = analisador.calcular_fft_2d(imagem_ruido_branco)

        razao_gradiente = analisador.calcular_razao_frequencia(espectro_gradiente)
        razao_ruido = analisador.calcular_razao_frequencia(espectro_ruido)

        assert razao_gradiente < razao_ruido, (
            f"Gradiente suave (razao={razao_gradiente:.4f}) deveria ter menor "
            f"razao HF/LF do que ruido branco (razao={razao_ruido:.4f})"
        )
