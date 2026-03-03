"""
Detector de Conteudo Gerado por IA
Entry point da aplicacao Streamlit.

Uso:
    streamlit run app.py
"""

import logging
import sys
from pathlib import Path

# Adiciona raiz do projeto ao path para que todos os modulos internos
# sejam importaveis sem instalacao do pacote.
sys.path.insert(0, str(Path(__file__).parent))

# Configuracao de logging antes de qualquer import interno,
# para capturar mensagens de inicializacao dos modulos.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

from visualization.dashboard import DashboardDeteccao  # noqa: E402


def main() -> None:
    """
    Ponto de entrada da aplicacao.

    Instancia e executa o dashboard principal. O Streamlit re-executa
    este script a cada interacao do usuario; o estado persistente e
    mantido via st.session_state e st.cache_resource.
    """
    logger.info("Iniciando Detector de Conteudo Gerado por IA...")
    dashboard = DashboardDeteccao()
    dashboard.renderizar()


if __name__ == "__main__":
    main()
