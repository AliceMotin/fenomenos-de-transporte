from PySide6 import QtWidgets, QtCore
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTextEdit, QComboBox, QTableWidget, QTableWidgetItem,
    QMessageBox, QFileDialog, QDialog, QFormLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import math
import numpy as np
import csv
import sys

# Aceleração da gravidade
g = 9.80665

# ----------------------------
# Funções Hidráulicas
# ----------------------------
def viscosidade_cinematica(viscosidade, densidade):
    """Calcula a viscosidade cinemática (nu = mu / rho)."""
    return viscosidade / densidade

def numero_reynolds(densidade, velocidade, diametro, viscosidade=None, visco_cinematica=None):
    """Calcula o Número de Reynolds (Re)."""
    if visco_cinematica is None:
        if viscosidade is None:
            raise ValueError("Forneça a viscosidade (mu) ou a viscosidade cinemática (nu).")
        visco_cinematica = viscosidade_cinematica(viscosidade, densidade)
    return velocidade * diametro / visco_cinematica if visco_cinematica != 0 else 0.0

def atrito_haaland(Re, eps_rel):
    """Calcula o fator de atrito 'f' pela fórmula de Haaland."""
    
    # Tratamento de erro (Re <= 0)
    if Re <= 0:
        return float('inf')
        
    # 1. REGIME LAMINAR (Re < 2000)
    if Re <= 2100:
        return 64.0 / Re
        
    # 2. REGIME TURBULENTO (Re >= 4000)
    # A fórmula de Haaland é mais precisa no regime plenamente turbulento.
    if Re >= 4000:
        A = (eps_rel / 3.7) ** 1.11
        B = 6.9 / Re
        # Fórmula de Haaland (resolvendo para f = (1.8*log...)^-2)
        f = (-1.8 * math.log10(A + B)) ** -2
        return f
        
    # 3. REGIME DE TRANSIÇÃO (2100 <= Re < 4000)
    # Usar a fórmula de Haaland (turbulenta) para a Transição (simplificação):
    A = (eps_rel / 3.7) ** 1.11
    B = 6.9 / Re
    f = (-1.8 * math.log10(A + B)) ** -2
    return f

def atrito_colebrook(Re, eps_rel, tol=1e-6, maxit=200):
    """Calcula o fator de atrito 'f' pela equação de Colebrook (método iterativo)."""
    if Re < 2100: # Regime laminar
        return 64.0 / Re
    # Inicializa com Haaland
    f = atrito_haaland(Re, eps_rel)
    for i in range(maxit):
        lhs = 1.0 / math.sqrt(f)
        rhs = -2.0 * math.log10(eps_rel/3.7 + 2.51/(Re*math.sqrt(f)))
        resid = lhs - rhs
        if abs(resid) < tol:
            return f
        # Método de Newton-Raphson (aproximação da derivada)
        df = f * 1e-6 if f > 0 else 1e-6
        f2 = f + df
        lhs2 = 1.0 / math.sqrt(f2)
        rhs2 = -2.0 * math.log10(eps_rel/3.7 + 2.51/(Re*math.sqrt(f2)))
        resid2 = lhs2 - rhs2
        dRdf = (resid2 - resid) / df
        f = f - resid / dRdf
        if f <= 0:
            f = 1e-12
    return f

def perda_carga_darcy(Comprimento, Diametro, velocidade, fator_atrito):
    """Calcula a perda de carga distribuída (hf) pela equação de Darcy-Weisbach."""
    return fator_atrito * (Comprimento/Diametro) * velocidade**2 / (2.0 * g)

def perda_carga_localizada(K_acessorio, velocidade):
    """Calcula a perda de carga localizada (hL) através do coeficiente K."""
    return K_acessorio * velocidade**2 / (2.0 * g)

TABELA_K_PADRAO = {
    "cotovelo_90_curto": 0.9, 
    "cotovelo_90_longo": 0.4, 
    "valvula_esfera_aberta": 0.05, 
    "entrada_tubo": 0.5, 
    "saida_tubo": 1.0
}

TABELA_K_DESCRICOES = {
    "cotovelo_90_curto": "Cotovelo 90° Curto",
    "cotovelo_90_longo": "Cotovelo 90° Longo",
    "valvula_esfera_aberta": "Válvula de Esfera (aberta)",
    "entrada_tubo": "Entrada no reservatório",
    "saida_tubo": "Saída para o reservatório"
}

# ----------------------------
# Container de Dados
# ----------------------------
class Trecho:
    """Representa um segmento de tubulação."""
    def __init__(self, nome="Trecho", C=10.0, D=0.05, rugosidade=1.5e-6, acessorios=None, delta_cota=0.0):
        self.nome = nome
        self.C = float(C) # Comprimento
        self.D = float(D) # Diâmetro
        self.rugosidade = float(rugosidade)
        self.acessorios = acessorios or []  # lista de (chave, quantidade)
        self.delta_cota = float(delta_cota)
        
    def K_total(self):
        """Calcula o coeficiente de perda total (K_total) para o trecho."""
        soma_K = 0.0
        for chave, quantidade in self.acessorios:
            soma_K += TABELA_K_PADRAO.get(chave, 0.0) * quantidade
        return soma_K

# ----------------------------
# Canvas Matplotlib
# ----------------------------
class CanvasMatplotlib(FigureCanvas):
    def __init__(self, parent=None, dpi=100):
        fig = Figure(dpi=dpi, figsize=(5,5))
        self.eixo = fig.add_subplot(111)
        super().__init__(fig)
        fig.tight_layout()


class DialogoAdicionarTrecho(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adicionar Trecho de Tubulação")
        self.setModal(True)
        
        # O layout principal vai conter o formulário e a tabela de referência
        layout_principal = QVBoxLayout(self) 

        # 1. Formulário de Dados do Trecho
        layout_form = QFormLayout()

        self.nome = QLineEdit("Trecho")
        self.comprimento = QLineEdit("10.0")
        self.diametro = QLineEdit("0.05")
        self.rugosidade = QLineEdit("1.5e-6")
        self.delta_cota = QLineEdit("0.0")
        self.acessorios = QLineEdit("cotovelo_90_curto,1") # Exemplo de uso

        layout_form.addRow("Nome:", self.nome)
        layout_form.addRow("Comprimento C (m):", self.comprimento)
        layout_form.addRow("Diâmetro D (m):", self.diametro)
        layout_form.addRow("Rugosidade (m):", self.rugosidade)
        layout_form.addRow("Δ Cota (m):", self.delta_cota)
        layout_form.addRow("Acessórios (chave,qtd;...):", self.acessorios)

        layout_principal.addLayout(layout_form)
        
        # 2. Tabela de Referência de Coeficientes K (Melhoria de Usabilidade)
        tabela_k = QTableWidget(len(TABELA_K_PADRAO), 2)
        tabela_k.setHorizontalHeaderLabels(["Chave (para o campo)", "K Unitário"])
        tabela_k.setEditTriggers(QTableWidget.NoEditTriggers) # Torna a tabela somente leitura
        
        row = 0
        for chave, k_valor in TABELA_K_PADRAO.items():
            item_chave = QTableWidgetItem(chave) 
            item_k = QTableWidgetItem(f"{k_valor}")
            
            tabela_k.setItem(row, 0, item_chave)
            tabela_k.setItem(row, 1, item_k)
            row += 1
            
        # --- CORREÇÃO DA LARGURA DA COLUNA ---
        # Faz a coluna 0 (Chave) se ajustar automaticamente ao conteúdo.
        tabela_k.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents) 
        
        # Mantém a última coluna (K Unitário) esticável para preencher o resto do espaço disponível.
        tabela_k.horizontalHeader().setStretchLastSection(True) 
        
        # Ajusta a altura para caber o conteúdo + cabeçalho
        tabela_k.setMaximumHeight(tabela_k.verticalHeader().length() + tabela_k.horizontalHeader().height() + 5)
        
        layout_principal.addWidget(QLabel("<b>Tabela de Coeficientes K Disponíveis:</b>"))
        layout_principal.addWidget(tabela_k)


        # 3. Botões
        botoes = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        botoes.accepted.connect(self.accept)
        botoes.rejected.connect(self.reject)
        layout_principal.addWidget(botoes)

    def obter_trecho(self):
        nome = self.nome.text().strip() or "Trecho"
        try:
            C = float(self.comprimento.text()); D = float(self.diametro.text()); rug = float(self.rugosidade.text()); elev = float(self.delta_cota.text())
        except:
            raise ValueError("Entradas numéricas inválidas no trecho")
        acessorios_raw = self.acessorios.text().strip()
        lista_acessorios = []
        if acessorios_raw:
            partes = acessorios_raw.split(";")
            for p in partes:
                if not p.strip(): continue
                try:
                    chave, qtd_str = p.split(",") 
                    lista_acessorios.append((chave.strip(), int(qtd_str.strip())))
                except:
                    continue # Ignora partes mal formatadas
        return Trecho(nome=nome, C=C, D=D, rugosidade=rug, acessorios=lista_acessorios, delta_cota=elev)
# ----------------------------
# Janela Principal da Aplicação
# ----------------------------
class JanelaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Perdas de Carga em Tubulações")
        self.resize(1200, 780)
        self._trechos = []
        self._ultimos_resultados = None
        self._configurar_pipeline_padrao()
        self._construir_interface()

    def _configurar_pipeline_padrao(self):
        """Inicializa as variáveis de estado (sem adicionar trechos de exemplo)."""

        self._trechos = [] 
        # Propriedades do fluido padrão
        self.densidade = 998.2  # kg/m³ (água a 20°C)
        self.viscosidade = 1.002e-3  # Pa·s (água a 20°C)

    def _construir_interface(self):
        central = QWidget(); self.setCentralWidget(central)
        layout_principal = QVBoxLayout(central)

        abas = QtWidgets.QTabWidget(); layout_principal.addWidget(abas)

        aba_pipeline = QWidget(); abas.addTab(aba_pipeline, "Tubulação"); self._construir_aba_pipeline(aba_pipeline)
        aba_resultados = QWidget(); abas.addTab(aba_resultados, "Resultados / Gráficos"); self._construir_aba_resultados(aba_resultados)
        aba_cav = QWidget(); abas.addTab(aba_cav, "Cavitação / NPSH"); self._construir_aba_cavitacao(aba_cav)
        aba_moody = QWidget(); abas.addTab(aba_moody, "Diagrama de Moody"); self._construir_aba_moody(aba_moody)
        aba_calc = QWidget(); abas.addTab(aba_calc, "Calculadora Hidráulica"); self._construir_aba_calculadora(aba_calc)

        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(140); layout_principal.addWidget(self.log)

    # ----------------------------
    # Aba Tubulação
    # ----------------------------
    def _construir_aba_pipeline(self, pai):
        layout = QHBoxLayout(pai)
        layout_esq = QVBoxLayout(); layout_dir = QVBoxLayout()
        layout.addLayout(layout_esq, 2); layout.addLayout(layout_dir, 3)

        self.tabela_trechos = QTableWidget(0, 6)
        self.tabela_trechos.setHorizontalHeaderLabels(["Nome", "C (m)", "D (m)", "Rug (m)", "ΔCota (m)", "Acessórios"])
        layout_esq.addWidget(self.tabela_trechos)
        self._atualizar_tabela_trechos()

        botoes_tabela = QHBoxLayout(); layout_esq.addLayout(botoes_tabela)
        btn_adicionar = QPushButton("Adicionar Trecho"); btn_adicionar.clicked.connect(self._dialogo_adicionar_trecho); botoes_tabela.addWidget(btn_adicionar)
        btn_remover = QPushButton("Remover Trecho"); btn_remover.clicked.connect(self._remover_trecho_selecionado); botoes_tabela.addWidget(btn_remover)

        # Controles do lado direito
        layout_dir.addWidget(QLabel("<b>Propriedades do Fluido</b>"))
        layout_fluido = QHBoxLayout(); layout_dir.addLayout(layout_fluido)
        lf = QVBoxLayout(); rf = QVBoxLayout(); layout_fluido.addLayout(lf); layout_fluido.addLayout(rf)

        lf.addWidget(QLabel("Densidade ρ (kg/m³):")); self.densidade_edit = QLineEdit(str(self.densidade)); lf.addWidget(self.densidade_edit)
        lf.addWidget(QLabel("Viscosidade μ (Pa·s):")); self.viscosidade_edit = QLineEdit(str(self.viscosidade)); lf.addWidget(self.viscosidade_edit)

        rf.addWidget(QLabel("Método f:")); self.metodo_f = QComboBox(); self.metodo_f.addItems(["Haaland","Colebrook"]); rf.addWidget(self.metodo_f)
        rf.addWidget(QLabel("Método Perda:")); self.metodo_perda = QComboBox(); self.metodo_perda.addItems(["Darcy-Weisbach"]); rf.addWidget(self.metodo_perda)

        layout_dir.addWidget(QLabel("<b>Condições Operacionais</b>"))
        qlay = QHBoxLayout(); layout_dir.addLayout(qlay); qlay.addWidget(QLabel("Q (L/s):")); self.vazao_edit = QLineEdit("10.0"); qlay.addWidget(self.vazao_edit)
        play = QHBoxLayout(); layout_dir.addLayout(play); play.addWidget(QLabel("Pressão Entrada (kPa):")); self.p_entrada_edit = QLineEdit(""); play.addWidget(self.p_entrada_edit)

        btn_calcular = QPushButton("Calcular Perfil e Perdas"); btn_calcular.clicked.connect(self._calcular_pipeline); layout_dir.addWidget(btn_calcular)

    def _atualizar_tabela_trechos(self):
        self.tabela_trechos.setRowCount(len(self._trechos))
        for i, t in enumerate(self._trechos):

            acessorios_str = ";".join([f"{chave},{qtd}" for chave,qtd in t.acessorios])
            items = [
                QTableWidgetItem(t.nome), QTableWidgetItem(str(t.C)),
                QTableWidgetItem(str(t.D)), QTableWidgetItem(str(t.rugosidade)),
                QTableWidgetItem(str(t.delta_cota)),
                QTableWidgetItem(acessorios_str)
            ]
            for j, item in enumerate(items):
                self.tabela_trechos.setItem(i, j, item)

    def _dialogo_adicionar_trecho(self):
        dlg = DialogoAdicionarTrecho(self)
        if dlg.exec() == QDialog.Accepted:
            try:
                trecho = dlg.obter_trecho()
                self._trechos.append(trecho)
                self._atualizar_tabela_trechos()
            except Exception as e:
                QMessageBox.warning(self, "Erro", f"Erro ao adicionar trecho: {e}")

    def _remover_trecho_selecionado(self):
        linha = self.tabela_trechos.currentRow()
        if linha >= 0 and linha < len(self._trechos):
            del self._trechos[linha]
            self._atualizar_tabela_trechos()

    def _atualizar_trechos_da_tabela(self):
        novos_trechos = []
        for i in range(self.tabela_trechos.rowCount()):
            try:
                # 1. Tenta ler todos os dados
                nome = self.tabela_trechos.item(i, 0).text()
                C = float(self.tabela_trechos.item(i, 1).text())
                D = float(self.tabela_trechos.item(i, 2).text())
                rug = float(self.tabela_trechos.item(i, 3).text())
                elev = float(self.tabela_trechos.item(i, 4).text())
                acessorios_raw = self.tabela_trechos.item(i, 5).text().strip() 

                # 2. Processa a string de Acessórios
                lista_acessorios = []
                if acessorios_raw:
                    partes = acessorios_raw.split(";")
                    for p in partes:
                        if not p.strip(): continue
                        try:
                            chave, qtd_str = p.split(",")
                            lista_acessorios.append((chave.strip(), int(qtd_str.strip())))
                        except:
                        
                            continue 
                
                # 3. Cria e armazena o novo objeto Trecho
                novos_trechos.append(Trecho(nome=nome, C=C, D=D, rugosidade=rug, acessorios=lista_acessorios, delta_cota=elev))
            
            except Exception as e:
                self.log.append(f"AVISO: A linha {i+1} da tabela não foi processada corretamente: {e}")
                continue
            
        self._trechos = novos_trechos
        if novos_trechos:
             self._atualizar_tabela_trechos()
             pass
        else:
             pass

    # ----------------------------
    # Aba Resultados / Gráficos
    # ----------------------------
    def _construir_aba_resultados(self, pai):
        layout = QVBoxLayout(pai)
        topo = QHBoxLayout(); layout.addLayout(topo)
        layout_esq = QVBoxLayout(); layout_dir = QVBoxLayout(); topo.addLayout(layout_esq, 2); topo.addLayout(layout_dir, 3)
        self.texto_resultados = QTextEdit(); self.texto_resultados.setReadOnly(True); layout_esq.addWidget(self.texto_resultados)
        self.canvas = CanvasMatplotlib(self); layout_dir.addWidget(self.canvas)
        rodape = QHBoxLayout(); layout.addLayout(rodape)
        btn_plotar = QPushButton("Atualizar Gráfico"); btn_plotar.clicked.connect(self._desenhar_grafico_piezometrico); rodape.addWidget(btn_plotar)
        btn_limpar = QPushButton("Limpar"); btn_limpar.clicked.connect(self._limpar_resultados); rodape.addWidget(btn_limpar)

    def _limpar_resultados(self):
        self.texto_resultados.clear(); self.canvas.eixo.clear(); self.canvas.draw(); self.log.append("Resultados limpos.")

    # ----------------------------
    # Aba Cavitação / NPSH
    # ----------------------------
    def _construir_aba_cavitacao(self, pai):
        layout = QVBoxLayout(pai)
        formulario = QHBoxLayout(); layout.addLayout(formulario)
        layout_esq = QVBoxLayout(); layout_dir = QVBoxLayout(); formulario.addLayout(layout_esq,2); formulario.addLayout(layout_dir,3)

        layout_esq.addWidget(QLabel("<b>Cálculo de Cavitação / NPSH</b>"))
        
        aviso_eps = QLabel("⚠ Aviso: Valores entre 1 e 100 são interpretados como Temperatura <br>e utilizados para calcular a Pv via Fórmula de Antoine")
        aviso_eps.setStyleSheet("font-size: 14px; color: orange;")
        layout_esq.addWidget(aviso_eps)
        
        layout_esq.addWidget(QLabel("Temperatura (°C) ou Pv (kPa):"))
        self.temperatura_edit = QLineEdit("20.0"); layout_esq.addWidget(self.temperatura_edit)
        layout_esq.addWidget(QLabel("Índice Nó Sucção (0=entrada):")); self.no_sucao_idx = QLineEdit("0"); layout_esq.addWidget(self.no_sucao_idx)
        layout_esq.addWidget(QLabel("NPSHr da Bomba (m):")); self.npshr_edit = QLineEdit("3.0"); layout_esq.addWidget(self.npshr_edit)
        btn_calcular_npsh = QPushButton("Calcular NPSHa"); btn_calcular_npsh.clicked.connect(self._calcular_cavitacao); layout_esq.addWidget(btn_calcular_npsh)

        layout_dir.addWidget(QLabel("<b>Resultados</b>"))
        self.resultado_cav = QTextEdit(); self.resultado_cav.setMaximumHeight(100); self.resultado_cav.setReadOnly(True); layout_dir.addWidget(self.resultado_cav)
        self.canvas_cav = CanvasMatplotlib(self); layout_dir.addWidget(self.canvas_cav)
        
    # ----------------------------
    # Aba Diagrama de Moody
    # ----------------------------
    def _construir_aba_moody(self, pai):
        layout = QVBoxLayout(pai)
        topo = QHBoxLayout(); layout.addLayout(topo)
        layout_esq = QVBoxLayout(); layout_dir = QVBoxLayout(); topo.addLayout(layout_esq,2); topo.addLayout(layout_dir,3)
        layout_esq.addWidget(QLabel("<b>Diagrama de Moody</b>"))
        layout_esq.addWidget(QLabel("Re Mín / Máx:"))
        linha = QHBoxLayout(); self.re_min = QLineEdit("50"); self.re_max = QLineEdit("1e7"); linha.addWidget(self.re_min); linha.addWidget(self.re_max); layout_esq.addLayout(linha)
        layout_esq.addWidget(QLabel("ε/D (sep ,):")); self.lista_eps = QLineEdit("1e-6,1e-5,5e-5"); layout_esq.addWidget(self.lista_eps)
        
        # Aviso sobre qual curva será usada para o cálculo do ponto
        aviso_eps = QLabel("⚠ O cálculo do ponto (f) usa apenas o 1º valor de ε/D.")
        aviso_eps.setStyleSheet("font-size: 14px; color: orange;")
        layout_esq.addWidget(aviso_eps)
                
        # Campo de entrada para o Re calculado
        layout_esq.addWidget(QLabel("Re do Cálculo (Opcional):"))
        self.re_calculado = QLineEdit("") # Inicializa vazio
        layout_esq.addWidget(self.re_calculado)
        
        layout_esq.addWidget(QLabel("Plotar Colebrook?")); self.cb_colebrook = QComboBox(); self.cb_colebrook.addItems(["Sim","Não"]); layout_esq.addWidget(self.cb_colebrook)
        layout_esq.addWidget(QPushButton("Atualizar Moody", clicked=self._atualizar_grafico_moody))
        self.canvas_moody = CanvasMatplotlib(self); layout_dir.addWidget(self.canvas_moody)
        obs = QLabel("Observação: Haaland (linha sólida) e Colebrook (tracejada).")
        obs.setMaximumHeight(18)
        obs.setStyleSheet("font-size: 10px; color: gray;")
        layout.addWidget(obs)

    # ----------------------------
    # Aba Calculadora (Bernoulli)
    # ----------------------------
    def _construir_aba_calculadora(self, pai):
    
        layout = QVBoxLayout(pai)
        
        texto_explicativo = (
            "<b>Como Usar a Calculadora (Equação da Energia):</b><br>"
            "1. Insira os valores (Z, P, D, v) dos dois pontos.<br>"
            "2. Deixe <b>APENAS UM CAMPO VAZIO</b> (P1 ou P2) para o simulador calcular o valor desconhecido.<br>"
            "3. Se P1 e P2 forem preenchidos, o simulador faz uma <b>VERIFICAÇÃO</b> (Residual ≈ 0).<br>"
            "4. <b>K Total:</b> Coeficiente de Perda Localizada (Somatório de Ks) entre os pontos. Use 0 se for desprezível.<br>"
            "5. Caso preencha todos os campos a calculadora fará uma interpretação dos resultado do LHS (lado esquerdo), RHS (lado direito) e Residual = LHS - RHS.<br>"
        )
        label_explicacao = QLabel(texto_explicativo)
        label_explicacao.setWordWrap(True)
        layout.addWidget(label_explicacao)
        
        linha = QHBoxLayout() 
        layout.addLayout(linha)
        
        layout_esq = QVBoxLayout()
        layout_dir = QVBoxLayout()
        
        linha.addLayout(layout_esq)
        linha.addLayout(layout_dir)
        
        # Campos Ponto 1 (Lado Esquerdo)
        layout_esq.addWidget(QLabel("Ponto 1 - Z1 (m):")); self.Z1 = QLineEdit("10.0"); layout_esq.addWidget(self.Z1)
        layout_esq.addWidget(QLabel("P1 (kPa) - opcional:")); self.P1 = QLineEdit(""); layout_esq.addWidget(self.P1)
        layout_esq.addWidget(QLabel("D1 (m):")); self.D1 = QLineEdit("0.05"); layout_esq.addWidget(self.D1)
        layout_esq.addWidget(QLabel("v1 (m/s) - opcional:")); self.v1 = QLineEdit(""); layout_esq.addWidget(self.v1)
        
        # Campos Ponto 2 (Lado Direito)
        layout_dir.addWidget(QLabel("Ponto 2 - Z2 (m):")); self.Z2 = QLineEdit("8.0"); layout_dir.addWidget(self.Z2)
        layout_dir.addWidget(QLabel("P2 (kPa) - opcional:")); self.P2 = QLineEdit(""); layout_dir.addWidget(self.P2)
        layout_dir.addWidget(QLabel("D2 (m):")); self.D2 = QLineEdit("0.05"); layout_dir.addWidget(self.D2)
        layout_dir.addWidget(QLabel("v2 (m/s) - opcional:")); self.v2 = QLineEdit(""); layout_dir.addWidget(self.v2)
        
        layout.addWidget(QLabel("K Total (adimensional):")); self.K_total = QLineEdit("0.0"); layout.addWidget(self.K_total)
        btn_bernoulli = QPushButton("Calcular Bernoulli", clicked=self._calcular_bernoulli); layout.addWidget(btn_bernoulli)
        self.resultado_calc = QTextEdit(); self.resultado_calc.setReadOnly(True); layout.addWidget(self.resultado_calc)

    # ----------------------------
    # Calcular Pipeline
    # ----------------------------
    def _calcular_pipeline(self):
        self._atualizar_trechos_da_tabela() 
        
        if not self._trechos:
            QMessageBox.warning(self, "Erro", "Não há trechos de tubulação válidos para calcular. Adicione um trecho primeiro."); return
        
        try:
            densidade = float(self.densidade_edit.text()); viscosidade = float(self.viscosidade_edit.text())
        except:
            QMessageBox.warning(self, "Erro", "Densidade ou viscosidade inválida"); return
        try:
            Vazao_Ls = float(self.vazao_edit.text())
        except:
            QMessageBox.warning(self, "Erro", "Vazão inválida"); return
            
        Vazao_m3s = Vazao_Ls / 1000.0 # Conversão para m³/s
        
        usar_colebrook = (self.metodo_f.currentText().startswith("Colebrook"))
        
        resultados = []; pressoes = []; P_entrada = None
        
        if self.p_entrada_edit.text().strip() != "":
            try:
                P_entrada = float(self.p_entrada_edit.text()) * 1000.0 # Converter kPa para Pa
            except:
                QMessageBox.warning(self, "Erro", "Pressão de entrada inválida"); return
                
        P_atual = P_entrada; h_total_somada = 0.0
        
        for trecho in self._trechos:
            Area = math.pi * (trecho.D ** 2) / 4.0; velocidade = Vazao_m3s / Area
            Re = numero_reynolds(self.densidade, velocidade, trecho.D, viscosidade=viscosidade)
            eps_rel = trecho.rugosidade / trecho.D
            
            fator_atrito = atrito_colebrook(Re, eps_rel) if usar_colebrook else atrito_haaland(Re, eps_rel)
            h_atrito = perda_carga_darcy(trecho.C, trecho.D, velocidade, fator_atrito)
                
            K_total = trecho.K_total()
            h_local = perda_carga_localizada(K_total, velocidade)
            h_total = h_atrito + h_local
            
            h_total_somada += h_total
            delta_p = self.densidade * g * h_total
            
            resultados.append({
                "segmento": trecho.nome, "C": trecho.C, "D": trecho.D, "v": velocidade, "Re": Re, "f": fator_atrito,
                "h_atrito": h_atrito, "K_total": K_total, "h_local": h_local, "h_total": h_total,
                "delta_p_Pa": delta_p, "delta_cota": trecho.delta_cota
            })
            
            if P_atual is not None:
                # P_saida = P_entrada - (perda distribuída + perda localizada + perda por elevação)
                P_atual = P_atual - delta_p - self.densidade * g * trecho.delta_cota
                pressoes.append(P_atual)

        # Output
        self.texto_resultados.clear()
        if P_entrada is not None:
            self.texto_resultados.append(f"Pressão de entrada: {P_entrada/1000.0:.3f} kPa\n")
            self.texto_resultados.append("Pressões nos nós (kPa):")
            lista_p = [P_entrada] + pressoes
            for i, pval in enumerate(lista_p):
                self.texto_resultados.append(f"Nó {i}: {pval/1000.0:.3f} kPa")
        else:
            self.texto_resultados.append(f"Perda de carga total: {h_total_somada:.6f} m (equivalente a {self.densidade*g*h_total_somada:.2f} Pa)\n")

        self.texto_resultados.append("\nDetalhes por trecho:")
        for r in resultados:
            linha_res = (f"{r['segmento']}: v={r['v']:.4f} m/s, Re={r['Re']:.1f}, "
                    f"f={'{:.5f}'.format(r['f']) if r['f'] else 'n/a'}, "
                    f"h_atrito={r['h_atrito']:.4f} m, Ktot={r['K_total']:.3f}, h_local={r['h_local']:.4f} m, h_total={r['h_total']:.4f} m")
            self.texto_resultados.append(linha_res)

        self._ultimos_resultados = {"segmentos": resultados, "pressoes": pressoes, "P_entrada": P_entrada}
        self.log.append("Cálculo da tubulação executado com sucesso.")
        self._desenhar_grafico_piezometrico()

    # ----------------------------
    # Desenhar Linhas Piezométrica e Geométrica
    # ----------------------------
    def _desenhar_grafico_piezometrico(self):
        if not self._ultimos_resultados or not self._ultimos_resultados["segmentos"]:
            # Não exibe erro, apenas limpa e informa no log se não houver dados para plotar.
            self.canvas.eixo.clear(); self.canvas.draw()
            self.log.append("AVISO: Nenhum trecho para plotar o perfil piezométrico.")
            return
            
        res = self._ultimos_resultados["segmentos"]; P_entrada = self._ultimos_resultados.get("P_entrada", None)
        
        # --------------------------------------------------------
        # VERIFICAÇÃO DE PRESSÃO DE ENTRADA ABSOLUTA 
        # --------------------------------------------------------
        
        # Verifica se a P_entrada não foi fornecida (é None) ou se é zero/negativa.
        # O P_entrada é crucial para estabelecer a Linha Piezométrica Absoluta.
        if P_entrada is None or P_entrada <= 0.0:
            self.canvas.eixo.clear()
            self.canvas.draw()
            self.log.append("ERRO: Não é possível plotar a Linha Piezométrica. P_entrada deve ser um valor absoluto positivo (> 0 kPa).")
            QMessageBox.warning(self, "Erro de Plotagem", 
                                "Pressão de Entrada inválida (Ausente ou <= 0 kPa). Plotagem da Linha Piezométrica Absoluta cancelada.")
            return # Interrompe a função se a pressão for inválida
        
        posicoes_C = [0.0]; carga_piez = []; carga_energia = []
        
        Carga_P = P_entrada / (self.densidade * g)
            
        carga_piez.append(Carga_P)
        carga_energia.append(Carga_P + 0.5 * (res[0]['v']**2) / g if res else Carga_P)
        
        for r in res:
            posicoes_C.append(posicoes_C[-1] + r['C'])
            # Carga Piezométrica = Carga Piezométrica anterior - Perda de Carga Total - Delta Cota
            Carga_P = Carga_P - r['h_total'] - r['delta_cota']
            carga_piez.append(Carga_P)
            # A velocidade é a mesma para todo o trecho, então a Carga Cinética é constante dentro do trecho
            carga_energia.append(Carga_P + 0.5 * (r['v']**2) / g)
            
        # Cota Geométrica (elevação acumulada)
        cota_nos = [0.0]; cota_acumulada = 0.0
        for r in res:
            cota_acumulada += r['delta_cota']; cota_nos.append(cota_acumulada)
            
        # Plotar
        self.canvas.eixo.clear()
        self.canvas.eixo.plot(posicoes_C, cota_nos, label='Tubulação (cota)', linewidth=2)
        self.canvas.eixo.plot(posicoes_C, carga_piez, marker='o', label='Linha Piezométrica')
        self.canvas.eixo.plot(posicoes_C, carga_energia, marker='s', label='Linha de Energia')
        self.canvas.eixo.set_xlabel("Posição ao longo da tubulação (m)")
        self.canvas.eixo.set_ylabel("Carga (m)")
        self.canvas.eixo.grid(True)
        self.canvas.eixo.legend()
        self.canvas.draw()
        self.log.append("Gráfico piezométrico atualizado.")


    # ----------------------------
    # Cavitação / NPSH
    # ----------------------------
    def _pv_de_temperatura_aprox(self, Temp_C):
        """Calcula a pressão de vapor Pv em kPa usando fórmula de Antoine (aprox)."""
        A, B, C = 8.07131, 1730.63, 233.426
        Pv_mmHg = 10 ** (A - B / (C + Temp_C))
        return Pv_mmHg * 0.133322 # Conversão de mmHg para kPa

    def _calcular_cavitacao(self):
        if not self._ultimos_resultados:
            QMessageBox.information(self, "Info", "Execute um cálculo na aba Tubulação primeiro."); return
            
        try:
            densidade = float(self.densidade_edit.text())
        except:
            QMessageBox.warning(self, "Erro", "Densidade inválida"); return
            
        temp_ou_pv = self.temperatura_edit.text().strip()
        
        try:
            val = float(temp_ou_pv)
            # Heurística: se entre 0 e 100 -> trata como temperatura (°C), senão trata como Pv (kPa)
            if 0 <= val <= 100:
                Pv_kPa = self._pv_de_temperatura_aprox(val)
            else:
                Pv_kPa = val
        except:
            Pv_kPa = 2.34 # Pv da água a 20°C (aprox)
            
        Pv_Pa = Pv_kPa * 1000.0
        
        try:
            idx_no = int(self.no_sucao_idx.text())
        except:
            QMessageBox.warning(self, "Erro", "Índice de nó inválido"); return
            
        if self._ultimos_resultados.get("P_entrada", None) is None:
            QMessageBox.information(self, "Info", "Forneça pressão de entrada na aba Tubulação para cálculo de NPSHa."); return
            
        pressoes = [self._ultimos_resultados["P_entrada"]] + self._ultimos_resultados["pressoes"]
        segmentos = self._ultimos_resultados["segmentos"]
        
        if idx_no < 0 or idx_no >= len(pressoes):
            QMessageBox.warning(self, "Erro", "Índice de nó fora do intervalo."); return
            
        # Obter cotas e velocidades nos nós (assumindo que a velocidade do segmento se estende até o nó de saída)
        cota_nos = [0.0]; cota_acumulada = 0.0
        for s in segmentos:
            cota_acumulada += s["delta_cota"]; cota_nos.append(cota_acumulada)
            
        velocidades_nos = [seg["v"] for seg in segmentos]; velocidades_nos.append(segmentos[-1]["v"]) # Velocidade na saída do último trecho
        
        P_no = pressoes[idx_no]; cota_no = cota_nos[idx_no]
        velocidade_no = velocidades_nos[idx_no] if idx_no < len(velocidades_nos) else velocidades_nos[-1]
        
        # NPSHa = (P/rho*g) + Z - (Pv/rho*g) - (v^2/2g)
        NPSHa = (P_no / (densidade * g)) + cota_no - (Pv_Pa / (densidade * g)) + (velocidade_no**2) / (2.0 * g)
        
        # Análise de cavitação em qualquer ponto
        p_min = min(pressoes); idx_min = pressoes.index(p_min); cota_min = cota_nos[idx_min]
        carga_min = p_min / (densidade * g) + cota_min
        carga_pv = Pv_Pa / (densidade * g)
        
        try:
            NPSHr = float(self.npshr_edit.text())
        except:
            NPSHr = 0.0
            
        txt = []
        txt.append(f"Pv usada: {Pv_kPa:.3f} kPa ({carga_pv:.4f} m de coluna de água)")
        txt.append(f"Nó escolhido = {idx_no}; NPSHa = {NPSHa:.4f} m; NPSHr = {NPSHr:.4f} m")
        txt.append(f"Pressão mínima: nó {idx_min} -> Pmin = {p_min/1000.0:.3f} kPa; carga = {carga_min:.4f} m")
        
        if NPSHa < NPSHr:
            txt.append("**NPSHa < NPSHr -> RISCO DE CAVITAÇÃO NA BOMBA!**")
        else:
            txt.append("**NPSHa >= NPSHr**")
            
        if carga_min <= carga_pv:
            txt.append("**Carga Mínima <= Pv -> RISCO DE CAVITAÇÃO AO LONGO DA TUBULAÇÃO!**")
        else:
            txt.append("OK: Carga Mínima > Pv")
            
        if NPSHa < NPSHr and carga_min <= carga_pv:
            txt.append("ALERTA MÁXIMO: Risco de cavitação grave e total!")
            
        self.resultado_cav.setPlainText("\n".join(txt))
        
        # Plot
        self.canvas_cav.eixo.clear()
        indices = list(range(len(pressoes)))
        cargas = [Pn / (densidade * g) + cn for Pn, cn in zip(pressoes, cota_nos)] # Carga Piezométrica + Cota
        
        self.canvas_cav.eixo.plot(indices, cargas, marker='o', label='Linha Piezométrica + Cota (m)')
        self.canvas_cav.eixo.axhline(carga_pv + cota_no, color='r', linestyle='--', label='Pv (m) no nó da bomba')
        self.canvas_cav.eixo.plot(idx_no, cargas[idx_no], 'ro', label='Ponto de Sucção')
        
        self.canvas_cav.eixo.set_xlabel("Nó (0 = entrada)")
        self.canvas_cav.eixo.set_ylabel("Carga (m)")
        self.canvas_cav.eixo.grid(True)
        self.canvas_cav.eixo.legend()
        self.canvas_cav.draw()
        self.log.append("Cálculo de cavitação realizado.")

    # ----------------------------
    # Diagrama de Moody
    # ----------------------------
    def _atualizar_grafico_moody(self):

        try:
            Re_min = float(self.re_min.text()); Re_max = float(self.re_max.text())
        except:
            QMessageBox.warning(self, "Erro", "Faixa de Re inválida"); return
            
        try:
            eps_vals = [float(x.strip()) for x in self.lista_eps.text().split(",") if x.strip()]
        except:
            QMessageBox.warning(self, "Erro", "Formato ε/D inválido"); return
            
        Re_valores = np.logspace(math.log10(max(1, Re_min)), math.log10(max(Re_min+1, Re_max)), 300)
        
        self.canvas_moody.eixo.clear()
        
        for eps in eps_vals:
            f_haaland = [atrito_haaland(Re, eps) if Re>0 else None for Re in Re_valores]
            self.canvas_moody.eixo.loglog(Re_valores, f_haaland, label=f'Haaland ε/D={eps:.0e}', linestyle='-')
            
        if self.cb_colebrook.currentText().startswith("Sim"):
            for eps in eps_vals:
                f_cole = [atrito_colebrook(Re, eps) if Re>0 else None for Re in Re_valores]
                self.canvas_moody.eixo.loglog(Re_valores, f_cole, label=f'Colebrook ε/D={eps:.0e}', linestyle='--')
                
        Re_lam = np.array([1, 2100]); f_lam = 64.0 / Re_lam
        self.canvas_moody.eixo.loglog(Re_lam, f_lam, 'k:', label='Laminar 64/Re')
        
        
        
    # ----------------------------------------------------------------------
    # Desenho do Ponto de Cálculo inserido pelo usuário
    # ----------------------------------------------------------------------
        Re_calc_str = self.re_calculado.text()

        if Re_calc_str and eps_vals:
            try:
                Re_calc = float(Re_calc_str)
                
                # Assume a primeira curva de rugosidade para traçar o ponto de interesse
                eps_principal = eps_vals[0] 
                
                # 1. Calcula o Fator de Atrito (f) para o Re inserido
                # Usa Colebrook se estiver selecionada, caso contrário usa Haaland
                if self.cb_colebrook.currentText().startswith("Sim"):
                    f_calc = atrito_colebrook(Re_calc, eps_principal)
                else:
                    f_calc = atrito_haaland(Re_calc, eps_principal)

                # 2. Desenha a linha vertical (Re constante)
                self.canvas_moody.eixo.axvline(x=Re_calc, color='r', linestyle='-.', linewidth=1.5, label=f'Re={Re_calc:.0f}')

                # 3. Desenha a linha horizontal (f constante)
                self.canvas_moody.eixo.loglog([Re_valores.min(), Re_calc], [f_calc, f_calc], 
                                            color='r', linestyle='-.', linewidth=1.5)
                
                # 4. Adiciona um marcador (ponto) no local de intersecção
                self.canvas_moody.eixo.plot(Re_calc, f_calc, 'ro', markersize=6, label=f'f={f_calc:.4f} (ε/D={eps_principal:.0e})')

                self.log.append(f"Ponto de cálculo traçado: Re={Re_calc:.0f}, f={f_calc:.4f}")
                
            except ValueError:
                QMessageBox.warning(self, "Erro", "Re de Cálculo inválido")
                pass
    # ----------------------------------------------------------------------
        
        self.canvas_moody.eixo.set_xlabel("Número de Reynolds (Re)", fontsize=10)
        self.canvas_moody.eixo.set_ylabel("Fator de atrito f", fontsize=10)
        self.canvas_moody.eixo.set_title("Diagrama de Moody", fontsize=10)

        # Configurações de fonte dos eixos
        self.canvas_moody.eixo.tick_params(axis='both', which='major', labelsize=10)
        self.canvas_moody.eixo.tick_params(axis='both', which='minor', labelsize=8)

        self.canvas_moody.eixo.grid(True, which='both', linestyle=':', alpha=0.5)
        self.canvas_moody.eixo.legend(fontsize='small')
        self.canvas_moody.draw()
        self.log.append("Diagrama de Moody atualizado.")

    # ----------------------------
    # Calculadora Bernoulli
    # ----------------------------
    def _calcular_bernoulli(self):
        try:
            Z1 = float(self.Z1.text()); Z2 = float(self.Z2.text())
            P1_str = self.P1.text().strip(); P2_str = self.P2.text().strip()
            D1 = float(self.D1.text()); D2 = float(self.D2.text())
            v1_str = self.v1.text().strip(); v2_str = self.v2.text().strip()
            K_total = float(self.K_total.text())
        except Exception as e:
            QMessageBox.warning(self, "Erro", "Entrada inválida: " + str(e)); return
            
        v1 = float(v1_str) if v1_str else None; v2 = float(v2_str) if v2_str else None
        alpha = 1.0 # Coeficiente de correção de energia cinética (Turbulento)
        
        if P1_str and not P2_str:
            P1 = float(P1_str) * 1000.0
            
            if v1 is None and v2 is None:
                QMessageBox.warning(self, "Erro", "Forneça ao menos uma velocidade"); return
                
            # Equação da Continuidade: v1 = v2 * (A2/A1)
            if v1 is None and v2 is not None and D1>0 and D2>0:
                A1 = math.pi*D1**2/4.0; A2 = math.pi*D2**2/4.0; v1 = v2*(A2/A1)
            if v2 is None and v1 is not None and D1>0 and D2>0:
                A1 = math.pi*D1**2/4.0; A2 = math.pi*D2**2/4.0; v2 = v1*(A1/A2)
                
            # Perda localizada
            h_local = perda_carga_localizada(K_total, v2 if v2 is not None else v1)
            
            # Cálculo de P2 (h1 = h2 + hL)
            P2 = self.densidade * g * (Z1 - Z2 + P1/(self.densidade*g) + (alpha*v1**2 - alpha*v2**2)/(2*g) - h_local)
            
            self.resultado_calc.setPlainText(f"P2 = {P2/1000.0:.4f} kPa")
            self.log.append("Calculadora: P2 calculado"); return
            
        if P2_str and not P1_str:
            P2 = float(P2_str) * 1000.0
            
            if v1 is None and v2 is None:
                QMessageBox.warning(self, "Erro", "Forneça ao menos uma velocidade"); return
                
            if v1 is None and v2 is not None and D1>0 and D2>0:
                A1 = math.pi*D1**2/4.0; A2 = math.pi*D2**2/4.0; v1 = v2*(A2/A1)
            if v2 is None and v1 is not None and D1>0 and D2>0:
                A1 = math.pi*D1**2/4.0; A2 = math.pi*D2**2/4.0; v2 = v1*(A1/A2)
                
            h_local = perda_carga_localizada(K_total, v2 if v2 is not None else v1)
            
            # Cálculo de P1 (h1 = h2 + hL)
            P1 = self.densidade * g * (Z2 - Z1 + P2/(self.densidade*g) + (alpha*v2**2 - alpha*v1**2)/(2*g) + h_local)
            
            self.resultado_calc.setPlainText(f"P1 = {P1/1000.0:.4f} kPa")
            self.log.append("Calculadora: P1 calculado"); return
            
        if P1_str and P2_str:
            # Verificação (ambos conhecidos)
            P1 = float(P1_str) * 1000.0; P2 = float(P2_str) * 1000.0
            v1 = float(v1) if v1 else 0.0; v2 = float(v2) if v2 else 0.0
            
            h_local = perda_carga_localizada(K_total, v2 if v2 != 0 else v1)
            
            LHS = Z1 + P1/(self.densidade*g) + 0.5*v1**2/g
            RHS = Z2 + P2/(self.densidade*g) + 0.5*v2**2/g + h_local
            
            self.resultado_calc.setPlainText(f"LHS={LHS:.6f} m, RHS={RHS:.6f} m, Residual={LHS-RHS:.6e} m")
            self.log.append("Calculadora: verificação feita"); return
            
        QMessageBox.information(self, "Info", "Caso não coberto. Forneça P1 ou P2 e pelo menos uma velocidade para cálculo.")

# ----------------------------
# Main
# ----------------------------
def main():
    app = QApplication(sys.argv)
    janela = JanelaPrincipal(); janela.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
