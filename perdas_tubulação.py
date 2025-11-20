# perdas_tubulacao_gui.py
# Projeto: Estudo Computacional de Perdas de Carga em Tubulações
# GUI: PyQt5
# Dependências: PyQt5, matplotlib, numpy
# Execute: python perdas_tubulacao_gui.py

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTextEdit, QComboBox, QTableWidget, QTableWidgetItem,
    QMessageBox, QFileDialog, QSpinBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import math
import numpy as np
import csv
import sys

g = 9.80665

# ----------------------------
#  Hydraulic / Fluid routines
# ----------------------------

def kinematic_viscosity(mu, rho):
    """Mu (Pa.s) -> nu (m2/s)"""
    return mu / rho

def reynolds_number(rho, v, D, mu=None, nu=None):
    if nu is None:
        if mu is None:
            raise ValueError("Forneça mu ou nu")
        nu = kinematic_viscosity(mu, rho)
    return v * D / nu

def friction_haaland(Re, eps_rel):
    if Re <= 0:
        return float('inf')
    if Re < 2300:
        return 64.0 / Re
    # Haaland explicit
    A = (eps_rel / 3.7) ** 1.11
    B = 6.9 / Re
    f = (-1.8 * math.log10(A + B)) ** -2
    return f

def friction_colebrook(Re, eps_rel, tol=1e-6, maxit=200):
    # Colebrook iterative solver for f
    if Re < 2300:
        return 64.0 / Re
    # Initial guess from Haaland
    f = friction_haaland(Re, eps_rel)
    for i in range(maxit):
        # Colebrook eq: 1/sqrt(f) = -2*log10( eps/3.7 + 2.51/(Re*sqrt(f)) )
        lhs = 1.0 / math.sqrt(f)
        rhs = -2.0 * math.log10(eps_rel/3.7 + 2.51/(Re*math.sqrt(f)))
        resid = lhs - rhs
        if abs(resid) < tol:
            return f
        # numerical derivative dR/df approximated
        df = f * 1e-6 if f>0 else 1e-6
        f2 = f + df
        lhs2 = 1.0 / math.sqrt(f2)
        rhs2 = -2.0 * math.log10(eps_rel/3.7 + 2.51/(Re*math.sqrt(f2)))
        resid2 = lhs2 - rhs2
        dRdf = (resid2 - resid) / df
        # Newton step
        f = f - resid / dRdf
        if f <= 0:
            f = 1e-12
    return f  # last value

def head_loss_darcy(L, D, v, f):
    return f * (L/D) * v**2 / (2.0 * g)

def head_loss_hazen_williams(L, D, Q, C):
    # units: m, m, m3/s, C (dimensionless)
    # formula: ΔH = 10.65 * L / D^4.87 * (Q/C)^1.85   (consistent with PDF)
    return 10.65 * L / (D ** 4.87) * (Q / C) ** 1.85

def minor_loss_head(K, v):
    return K * v**2 / (2.0 * g)

# Default Hazen-Williams C table (from PDF Quadro 2)
HAZEN_WILLIAMS_C_TABLE = {
    "Aço galvanizado": 125,
    "Aço sem costura novo": 130,
    "Aço soldado novo": 130,
    "Cobre/Latão": 130,
    "Ferro fundido novo": 130,
    "PVC": 150,
    "Concreto bom acabamento": 130,
    "Concreto comum": 120,
    "PEAD": 150,
    "Madeira": 120
}

# Default K table (typical values; can be edited)
DEFAULT_K_TABLE = {
    "elbow_90_short": 0.9,
    "elbow_90_long": 0.4,
    "tee_run": 1.8,
    "tee_branch": 3.0,
    "sudden_contraction": 0.5,
    "sudden_expansion": 0.2,
    "globe_valve_open": 10.0,
    "gate_valve_open": 0.2,
    "ball_valve_open": 0.05,
    "pipe_entry": 0.5,
    "pipe_exit": 1.0,
}

# ----------------------------
#  Data structures
# ----------------------------

class Segment:
    def __init__(self, name="Trecho", L=10.0, D=0.05, roughness=1.5e-6, fittings=None, elevation_delta=0.0):
        self.name = name
        self.L = L
        self.D = D
        self.roughness = roughness
        self.fittings = fittings or []  # list of (key, qty)
        self.elevation_delta = elevation_delta

    def total_K(self):
        s = 0.0
        for key, qty in self.fittings:
            s += DEFAULT_K_TABLE.get(key, 0.0) * qty
        return s

# ----------------------------
#  Three reservoirs solver
# ----------------------------

def three_reservoirs(Qguess_fun, bounds=(-1000, 1000), tol=1e-6, maxit=100):
    """
    Generic bisection root finder for the function Qguess_fun(X) that returns F(X) = Q1(X)-Q2(X)-Q3(X)
    X is the piezometric head at junction.
    """
    a, b = bounds
    fa = Qguess_fun(a)
    fb = Qguess_fun(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ValueError("Bisection: sinal igual nas bordas. Tente outros limites.")
    for i in range(maxit):
        m = 0.5 * (a + b)
        fm = Qguess_fun(m)
        if abs(fm) < tol:
            return m
        if fa * fm < 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
    return 0.5 * (a + b)

# ----------------------------
#  GUI: PyQt5 App
# ----------------------------

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, dpi=100):
        fig = Figure(dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        fig.tight_layout()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Estudo Computacional de Perdas de Carga - PyQt5")
        self.resize(1100, 700)
        self._segments = []
        self._setup_default_pipeline()
        self._build_ui()

    def _setup_default_pipeline(self):
        # Example: two segments default
        s1 = Segment(name="Trecho A", L=10.0, D=0.05, roughness=1.5e-6, fittings=[("elbow_90_short", 2)], elevation_delta=0.0)
        s2 = Segment(name="Trecho B", L=25.0, D=0.05, roughness=1.5e-6, fittings=[("ball_valve_open", 1)], elevation_delta=2.0)
        self._segments = [s1, s2]
        # Fluid properties default (water ~20C)
        self.rho = 998.2
        self.mu = 1.002e-3

    def _build_ui(self):
        central = QWidget()
        vlayout = QVBoxLayout()
        central.setLayout(vlayout)
        self.setCentralWidget(central)

        # Top controls: tabs
        tabs = QtWidgets.QTabWidget()
        vlayout.addWidget(tabs)

        # Tab: Pipeline editor
        tab_pipeline = QWidget()
        tabs.addTab(tab_pipeline, "Pipeline")
        self._build_pipeline_tab(tab_pipeline)

        # Tab: Results & Graph
        tab_results = QWidget()
        tabs.addTab(tab_results, "Resultados / Gráficos")
        self._build_results_tab(tab_results)

        # Tab: 3-Reservatórios
        tab_three = QWidget()
        tabs.addTab(tab_three, "3 Reservatórios")
        self._build_three_tab(tab_three)

        # Tab: Calculadora hidráulica
        tab_calc = QWidget()
        tabs.addTab(tab_calc, "Calculadora Hidráulica")
        self._build_calc_tab(tab_calc)

        # Bottom: log / messages
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        vlayout.addWidget(self.log)

    # ----------------------------
    # Pipeline tab build
    # ----------------------------
    def _build_pipeline_tab(self, parent):
        layout = QHBoxLayout()
        parent.setLayout(layout)

        left = QVBoxLayout()
        right = QVBoxLayout()
        layout.addLayout(left, 2)
        layout.addLayout(right, 3)

        # Segment table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Nome", "L (m)", "D (m)", "Rug (m)", "Cota Δ (m)", "Fittings"])
        left.addWidget(self.table)
        self._refresh_segment_table()

        btns = QHBoxLayout()
        left.addLayout(btns)
        add_btn = QPushButton("Adicionar trecho")
        add_btn.clicked.connect(self._add_segment_dialog)
        btns.addWidget(add_btn)
        del_btn = QPushButton("Remover trecho selecionado")
        del_btn.clicked.connect(self._remove_selected_segment)
        btns.addWidget(del_btn)

        load_btn = QPushButton("Carregar exemplo do PDF")
        load_btn.clicked.connect(self._load_pdf_example)
        btns.addWidget(load_btn)

        # Right: fluid & calculation controls
        right.addWidget(QLabel("<b>Propriedades do fluido</b>"))
        form = QHBoxLayout()
        right.addLayout(form)
        left_form = QVBoxLayout()
        right_form = QVBoxLayout()
        form.addLayout(left_form)
        form.addLayout(right_form)

        left_form.addWidget(QLabel("Densidade ρ (kg/m³):"))
        self.rho_edit = QLineEdit(str(self.rho))
        left_form.addWidget(self.rho_edit)
        left_form.addWidget(QLabel("Viscosidade μ (Pa·s):"))
        self.mu_edit = QLineEdit(str(self.mu))
        left_form.addWidget(self.mu_edit)

        right_form.addWidget(QLabel("Método fator f:"))
        self.f_method = QComboBox()
        self.f_method.addItems(["Haaland (rápido)", "Colebrook (iterativo)"])
        right_form.addWidget(self.f_method)

        right_form.addWidget(QLabel("Método perda de carga:"))
        self.loss_method = QComboBox()
        self.loss_method.addItems(["Darcy-Weisbach (universal)", "Hazen-Williams (empírico)"])
        right_form.addWidget(self.loss_method)

        # Q input and inlet pressure
        right.addWidget(QLabel("<b>Condições de contorno</b>"))
        q_layout = QHBoxLayout()
        right.addLayout(q_layout)
        q_layout.addWidget(QLabel("Vazão Q (L/s):"))
        self.q_edit = QLineEdit("10.0")
        q_layout.addWidget(self.q_edit)

        p_layout = QHBoxLayout()
        right.addLayout(p_layout)
        p_layout.addWidget(QLabel("Pressão entrada (kPa) — deixar vazio se não:"))
        self.p_in_edit = QLineEdit("")
        p_layout.addWidget(self.p_in_edit)

        # Buttons
        run_btn = QPushButton("Calcular perfil e perdas")
        run_btn.clicked.connect(self._compute_pipeline)
        right.addWidget(run_btn)

        exp_csv = QPushButton("Exportar resultados CSV")
        exp_csv.clicked.connect(self._export_csv)
        right.addWidget(exp_csv)

    def _refresh_segment_table(self):
        self.table.setRowCount(len(self._segments))
        for i, s in enumerate(self._segments):
            items = [
                QTableWidgetItem(s.name),
                QTableWidgetItem(str(s.L)),
                QTableWidgetItem(str(s.D)),
                QTableWidgetItem(str(s.roughness)),
                QTableWidgetItem(str(s.elevation_delta)),
                QTableWidgetItem(", ".join([f"{k}x{q}" for k, q in s.fittings]))
            ]
            for j, it in enumerate(items):
                self.table.setItem(i, j, it)

    def _add_segment_dialog(self):
        dlg = AddSegmentDialog(self)
        if dlg.exec_():
            seg = dlg.get_segment()
            self._segments.append(seg)
            self._refresh_segment_table()

    def _remove_selected_segment(self):
        row = self.table.currentRow()
        if row >= 0 and row < len(self._segments):
            del self._segments[row]
            self._refresh_segment_table()

    def _load_pdf_example(self):
        # Load the example from earlier: 2 segments with defaults
        self._setup_default_pipeline()
        self._refresh_segment_table()
        self.log.append("Exemplo padrão (do PDF) carregado.")

    # ----------------------------
    # Results tab build
    # ----------------------------
    def _build_results_tab(self, parent):
        layout = QVBoxLayout()
        parent.setLayout(layout)
        top = QHBoxLayout()
        layout.addLayout(top)
        left = QVBoxLayout()
        right = QVBoxLayout()
        top.addLayout(left, 2)
        top.addLayout(right, 3)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        left.addWidget(self.results_text)

        # Graph canvas
        self.canvas = MplCanvas(self)
        right.addWidget(self.canvas)

        # Buttons below
        btn_layout = QHBoxLayout()
        layout.addLayout(btn_layout)
        plot_btn = QPushButton("Atualizar gráfico (Linha piezométrica / energia)")
        plot_btn.clicked.connect(self._draw_piezometric_plot)
        btn_layout.addWidget(plot_btn)
        clear_btn = QPushButton("Limpar resultados")
        clear_btn.clicked.connect(self._clear_results)
        btn_layout.addWidget(clear_btn)

    def _clear_results(self):
        self.results_text.clear()
        self.canvas.ax.clear()
        self.canvas.draw()
        self.log.append("Resultados limpos.")

    # ----------------------------
    # 3 reservoirs tab build
    # ----------------------------
    def _build_three_tab(self, parent):
        layout = QVBoxLayout()
        parent.setLayout(layout)
        # We'll implement a simplified UI to input reservoir levels and pipe parameters
        form_layout = QHBoxLayout()
        layout.addLayout(form_layout)
        left = QVBoxLayout()
        right = QVBoxLayout()
        form_layout.addLayout(left)
        form_layout.addLayout(right)

        left.addWidget(QLabel("<b>Níveis (m)</b>"))
        self.na1 = QLineEdit("30.0")
        self.na2 = QLineEdit("24.0")
        self.na3 = QLineEdit("15.0")
        left.addWidget(QLabel("Reservatório 1 (NA1):")); left.addWidget(self.na1)
        left.addWidget(QLabel("Reservatório 2 (NA2):")); left.addWidget(self.na2)
        left.addWidget(QLabel("Reservatório 3 (NA3):")); left.addWidget(self.na3)

        right.addWidget(QLabel("<b>Trechos</b>"))
        # For each of 3 pipes: length (m), D (mm), C (H-W)
        self.t_L = [QLineEdit("1200"), QLineEdit("900"), QLineEdit("1500")]
        self.t_D = [QLineEdit("300"), QLineEdit("200"), QLineEdit("150")]
        self.t_C = [QLineEdit("90"), QLineEdit("120"), QLineEdit("125")]
        for i in range(3):
            right.addWidget(QLabel(f"Trecho {i+1} — L (m) / D (mm) / C:"))
            row = QHBoxLayout()
            row.addWidget(self.t_L[i]); row.addWidget(self.t_D[i]); row.addWidget(self.t_C[i])
            right.addLayout(row)

        # Buttons
        btn = QPushButton("Resolver problema dos 3 reservatórios (Hazen-Williams)")
        btn.clicked.connect(self._solve_three_reservoirs)
        layout.addWidget(btn)
        self.three_result = QTextEdit()
        self.three_result.setReadOnly(True)
        layout.addWidget(self.three_result)

    # ----------------------------
    # Calculator tab build
    # ----------------------------
    def _build_calc_tab(self, parent):
        layout = QVBoxLayout()
        parent.setLayout(layout)
        layout.addWidget(QLabel("<b>Calculadora Hidráulica (Bernoulli real)</b>"))
        row = QHBoxLayout()
        layout.addLayout(row)

        left = QVBoxLayout(); right = QVBoxLayout()
        row.addLayout(left); row.addLayout(right)

        left.addWidget(QLabel("Ponto 1 — Cota Z1 (m):")); self.Z1 = QLineEdit("10.0"); left.addWidget(self.Z1)
        left.addWidget(QLabel("P1 (kPa) — deixar vazio se desconhecido:")); self.P1 = QLineEdit(""); left.addWidget(self.P1)
        left.addWidget(QLabel("D1 (m):")); self.D1 = QLineEdit("0.05"); left.addWidget(self.D1)
        left.addWidget(QLabel("v1 (m/s) — opcional:")); self.v1 = QLineEdit(""); left.addWidget(self.v1)

        right.addWidget(QLabel("Ponto 2 — Cota Z2 (m):")); self.Z2 = QLineEdit("8.0"); right.addWidget(self.Z2)
        right.addWidget(QLabel("P2 (kPa) — deixar vazio se desconhecido:")); self.P2 = QLineEdit(""); right.addWidget(self.P2)
        right.addWidget(QLabel("D2 (m):")); self.D2 = QLineEdit("0.05"); right.addWidget(self.D2)
        right.addWidget(QLabel("v2 (m/s) — opcional:")); self.v2 = QLineEdit(""); right.addWidget(self.v2)

        # Other inputs
        layout.addWidget(QLabel("Perda localizada K total (soma) — incluir acessórios (valor adimensional):"))
        self.K_total = QLineEdit("0.0")
        layout.addWidget(self.K_total)

        # Buttons
        bt = QPushButton("Calcular (resolve incógnita se existir)")
        bt.clicked.connect(self._calculate_bernoulli)
        layout.addWidget(bt)
        self.calc_result = QTextEdit()
        self.calc_result.setReadOnly(True)
        layout.addWidget(self.calc_result)

    # ----------------------------
    # Actions: compute pipeline
    # ----------------------------
    def _compute_pipeline(self):
        try:
            rho = float(self.rho_edit.text())
            mu = float(self.mu_edit.text())
        except:
            QMessageBox.warning(self, "Erro", "Densidade ou viscosidade inválida")
            return
        try:
            Q_Ls = float(self.q_edit.text())
        except:
            QMessageBox.warning(self, "Erro", "Vazão inválida")
            return
        Q = Q_Ls / 1000.0  # L/s -> m3/s

        use_hw = (self.loss_method.currentText().startswith("Hazen"))
        use_colebrook = (self.f_method.currentText().startswith("Colebrook"))

        results = []
        pressures = []
        P = None
        if self.p_in_edit.text().strip() != "":
            try:
                P = float(self.p_in_edit.text()) * 1000.0  # kPa -> Pa
            except:
                QMessageBox.warning(self, "Erro", "Pressão de entrada inválida")
                return

        total_h_sum = 0.0
        P_current = P
        for seg in self._segments:
            A = math.pi * (seg.D ** 2) / 4.0
            v = Q / A
            Re = reynolds_number(rho, v, seg.D, mu=mu)
            eps_rel = seg.roughness / seg.D
            if use_hw:
                # Hazen-Williams needs C (we'll use heuristic default based on material PVC unless user changed)
                # We'll use 130 as default if no material specification
                C = 130.0
                try:
                    h_f = head_loss_hazen_williams(seg.L, seg.D, Q, C)
                except Exception as e:
                    h_f = 0.0
                f = None
            else:
                if use_colebrook:
                    f = friction_colebrook(Re, eps_rel)
                else:
                    f = friction_haaland(Re, eps_rel)
                h_f = head_loss_darcy(seg.L, seg.D, v, f)
            Ktot = seg.total_K()
            h_k = minor_loss_head(Ktot, v)
            total_h = h_f + h_k
            total_h_sum += total_h
            delta_p = rho * g * total_h  # Pa
            results.append({
                "segment": seg.name,
                "L": seg.L,
                "D": seg.D,
                "v": v,
                "Re": Re,
                "f": f,
                "h_f": h_f,
                "K_total": Ktot,
                "h_k": h_k,
                "h_total": total_h,
                "delta_p_Pa": delta_p,
                "elevation_delta": seg.elevation_delta
            })
            if P_current is not None:
                P_current = P_current - delta_p - rho * g * seg.elevation_delta
                pressures.append(P_current)

        # Output results
        self.results_text.clear()
        if P is not None:
            self.results_text.append(f"Pressão de entrada: {P/1000.0:.3f} kPa\n")
            self.results_text.append("Pressões nos nós (kPa):")
            p_list = [P] + pressures
            for i, pval in enumerate(p_list):
                self.results_text.append(f"Nó {i}: {pval/1000.0:.3f} kPa")
        else:
            self.results_text.append(f"Perda de carga total: {total_h_sum:.6f} m (equivalente a {rho*g*total_h_sum:.2f} Pa)\n")

        self.results_text.append("\nDetalhes por trecho:")
        for r in results:
            line = (f"{r['segment']}: v={r['v']:.4f} m/s, Re={r['Re']:.1f}, "
                    f"f={'{:.5f}'.format(r['f']) if r['f'] else 'n/a'}, "
                    f"h_f={r['h_f']:.4f} m, Ktot={r['K_total']:.3f}, h_k={r['h_k']:.4f} m, h_total={r['h_total']:.4f} m")
            self.results_text.append(line)

        # store last results for plotting/export
        self._last_results = {"segments": results, "pressures": pressures, "P_in": P}
        self.log.append("Cálculo executado com sucesso.")

        # draw plot
        self._draw_piezometric_plot()

    def _draw_piezometric_plot(self):
        if not hasattr(self, "_last_results"):
            QMessageBox.information(self, "Info", "Execute um cálculo primeiro (aba Pipeline).")
            return
        res = self._last_results["segments"]
        P_in = self._last_results.get("P_in", None)
        # build cumulative lengths and piezometric heads
        xs = [0.0]
        h_piez = []
        h_energy = []
        z = 0.0
        Hp = None
        # We'll create a synthetic piezometric head assuming inlet pressure known or zero baseline
        if P_in is None:
            # baseline zero: piezometric head equal to sum of losses backwards (relative plot)
            # compute piezometric head starting at 0 and increasing upstream
            Hp0 = 0.0
        else:
            Hp0 = P_in / (self.rho * g)
        Hp = Hp0
        h_piez.append(Hp)
        h_energy.append(Hp + 0.5 * (res[0]['v'] ** 2) / g if res else Hp)
        for r in res:
            xs.append(xs[-1] + r['L'])
            # subtract losses (piezometric goes down)
            Hp = Hp - r['h_total'] - r['elevation_delta']
            h_piez.append(Hp)
            h_energy.append(Hp + 0.5 * (r['v'] ** 2) / g)
        # Plot
        self.canvas.ax.clear()
        self.canvas.ax.plot(xs, h_piez, marker='o', label='Linha piezométrica')
        self.canvas.ax.plot(xs, h_energy, marker='s', label='Linha de energia')
        self.canvas.ax.set_xlabel("Posição ao longo da tubulação (m)")
        self.canvas.ax.set_ylabel("Carga (m)")
        self.canvas.ax.grid(True)
        self.canvas.ax.legend()
        self.canvas.draw()
        self.log.append("Gráfico atualizado.")

    def _export_csv(self):
        if not hasattr(self, "_last_results"):
            QMessageBox.information(self, "Info", "Execute um cálculo primeiro.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Salvar CSV", "", "CSV Files (*.csv)")
        if not fname:
            return
        res = self._last_results["segments"]
        with open(fname, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Segment", "L_m", "D_m", "v_m_s", "Re", "f", "h_f_m", "K_total", "h_k_m", "h_total_m", "delta_p_Pa"])
            for r in res:
                writer.writerow([r['segment'], r['L'], r['D'], r['v'], r['Re'], r['f'], r['h_f'], r['K_total'], r['h_k'], r['h_total'], r['delta_p_Pa']])
        self.log.append(f"Resultados exportados para {fname}")

    # ----------------------------
    # Three reservoirs solver action
    # ----------------------------
    def _solve_three_reservoirs(self):
        try:
            NA1 = float(self.na1.text())
            NA2 = float(self.na2.text())
            NA3 = float(self.na3.text())
            Ls = [float(self.t_L[i].text()) for i in range(3)]
            Ds = [float(self.t_D[i].text()) / 1000.0 for i in range(3)]
            Cs = [float(self.t_C[i].text()) for i in range(3)]
        except Exception as e:
            QMessageBox.warning(self, "Erro", "Entrada inválida: " + str(e))
            return

        # We will follow the approach from the PDF: assume HW formula for each pipe
        # Q_i = beta_i * (J_i)^(1/1.85) ... but easier: write Q = function(X) using Hazen-Williams
        # For each pipe between reservoir and junction with head X:
        # If pipe goes from reservoir at NH to junction X, the head loss unit J = (NH - X) / L
        # then Q = (J / beta)^(1/1.85)
        # beta depends on diameter and C -> from PDF beta table. We'll compute beta per formula:
        # From PDF: J = beta * Q^1.85  -> beta = J / Q^1.85. But we need a forward formula for Q.
        # Use Hazen-Williams rearranged: Q = C * ( D^2.63 * J^(1/1.85) / K1 )? To avoid confusion, we will directly solve numerically:
        # For a given X, Q1 = function(X): J1 = (NA1 - X)/L1 -> Q1 = sign* ( J1 / beta1 )^(1/1.85)
        # We can compute beta using D and C using the same units used in PDF. We'll use a conversion numeric approach:
        # We'll compute Q from Hazen-Williams direct formula invert via Q = C * ( (J * D^4.87) / 10.65 / L )^(1/1.85)

        def Q_from_HW(J, D, C, L):
            # J (m/m), D (m), C dimensionless, L (m)
            # From ΔH = 10.65 * L / D^4.87 * (Q/C)^1.85
            # => (Q/C)^1.85 = ΔH * D^4.87 / (10.65 * L)
            # => Q = C * [ ΔH * D^4.87 / (10.65 * L) ]^(1/1.85)
            if J <= 0:
                return 0.0
            term = J * (D ** 4.87) / (10.65 * L)
            if term <= 0:
                return 0.0
            return C * (term ** (1.0 / 1.85))

        # Define residual function F(X) = Q1(X) - Q2(X) - Q3(X)
        def residual(X):
            # Q1: from NA1 to X
            J1 = (NA1 - X) / Ls[0]
            Q1 = Q_from_HW(J1 if J1>0 else 0.0, Ds[0], Cs[0], Ls[0])
            # Q2: from X to NA2 (could be sign reversed)
            J2 = (X - NA2) / Ls[1]
            Q2 = Q_from_HW(J2 if J2>0 else 0.0, Ds[1], Cs[1], Ls[1])
            # Q3: from X to NA3
            J3 = (X - NA3) / Ls[2]
            Q3 = Q_from_HW(J3 if J3>0 else 0.0, Ds[2], Cs[2], Ls[2])
            # Flow sign convention: positive Q1 out of reservoir 1, Q2 out of junction to res2 etc.
            # This follows the PDF convention: Q1 = Q2 + Q3
            return Q1 - Q2 - Q3

        # Search for bracket for X between min(NA1,NA2,NA3)-50 and max +50
        lo = min(NA1, NA2, NA3) - 100.0
        hi = max(NA1, NA2, NA3) + 100.0
        try:
            X_sol = three_reservoirs(residual, bounds=(lo, hi), tol=1e-6, maxit=100)
        except Exception as e:
            QMessageBox.warning(self, "Erro", f"Não foi possível encontrar solução automaticamente: {e}")
            return

        # Compute flows
        def Q_at_X(X):
            J1 = (NA1 - X) / Ls[0]
            Q1 = Q_from_HW(J1 if J1>0 else 0.0, Ds[0], Cs[0], Ls[0])
            J2 = (X - NA2) / Ls[1]
            Q2 = Q_from_HW(J2 if J2>0 else 0.0, Ds[1], Cs[1], Ls[1])
            J3 = (X - NA3) / Ls[2]
            Q3 = Q_from_HW(J3 if J3>0 else 0.0, Ds[2], Cs[2], Ls[2])
            return Q1, Q2, Q3

        Q1, Q2, Q3 = Q_at_X(X_sol)
        self.three_result.clear()
        self.three_result.append(f"Solução: cota piezométrica no nó X = {X_sol:.6f} m")
        self.three_result.append(f"Vazões (m³/s): Q1 = {Q1:.6f}, Q2 = {Q2:.6f}, Q3 = {Q3:.6f}")
        self.three_result.append(f"Vazões (L/s): Q1 = {Q1*1000.0:.3f}, Q2 = {Q2*1000.0:.3f}, Q3 = {Q3*1000.0:.3f}")
        self.log.append("Problema dos 3 reservatórios resolvido (Hazen-Williams).")

    # ----------------------------
    # Calculator action
    # ----------------------------
    def _calculate_bernoulli(self):
        try:
            Z1 = float(self.Z1.text())
            Z2 = float(self.Z2.text())
            P1_str = self.P1.text().strip()
            P2_str = self.P2.text().strip()
            D1 = float(self.D1.text())
            D2 = float(self.D2.text())
            v1_str = self.v1.text().strip()
            v2_str = self.v2.text().strip()
            Ktot = float(self.K_total.text())
            rho = float(self.rho_edit.text())
        except Exception as e:
            QMessageBox.warning(self, "Erro", "Entrada inválida: " + str(e))
            return

        # Compute velocities if not provided using continuity? We don't have Q here.
        # If both v unknown -> problem underdetermined; we'll assume equal velocities if diameters equal and no Q.
        v1 = float(v1_str) if v1_str else None
        v2 = float(v2_str) if v2_str else None

        # beta term alpha assumed 1
        alpha = 1.0

        # Unknowns: either P1 or P2 missing -> solve for it. If both present -> just compute check.
        if P1_str and not P2_str:
            P1 = float(P1_str) * 1000.0
            # compute v1 or v2 if missing (can't deduce Q) -> if one missing and diameters equal assume conservation? We'll only solve simple cases.
            if v1 is None and v2 is None:
                QMessageBox.warning(self, "Erro", "Forneça ao menos uma velocidade quando P1 conhecido e P2 desconhecido.")
                return
            # compute missing velocity via continuity if possible
            if v1 is None and v2 is not None and D1>0 and D2>0:
                A1 = math.pi*D1**2/4.0
                A2 = math.pi*D2**2/4.0
                v1 = v2 * (A2/A1)
            if v2 is None and v1 is not None and D1>0 and D2>0:
                A1 = math.pi*D1**2/4.0
                A2 = math.pi*D2**2/4.0
                v2 = v1 * (A1/A2)
            # compute ht losses: h_k = K*(v^2/2g) using velocity at section where K applied: approximate with v2
            # For distributed losses (Darcy), user didn't input L and f here; we skip and only compute local loss contribution
            h_k = minor_loss_head(Ktot, v2 if v2 is not None else v1)
            # Bernoulli: Z1 + P1/(rho g) + v1^2/(2g) = Z2 + P2/(rho g) + v2^2/(2g) + h_k
            P2 = rho * g * (Z1 - Z2 + P1/(rho*g) + (alpha*v1**2 - alpha*v2**2)/(2*g) - h_k)
            self.calc_result.setPlainText(f"P2 = {P2/1000.0:.4f} kPa")
            self.log.append("Calculadora hidráulica calculou P2.")
            return

        if P2_str and not P1_str:
            P2 = float(P2_str) * 1000.0
            if v1 is None and v2 is None:
                QMessageBox.warning(self, "Erro", "Forneça ao menos uma velocidade quando P2 conhecido e P1 desconhecido.")
                return
            if v1 is None and v2 is not None and D1>0 and D2>0:
                A1 = math.pi*D1**2/4.0
                A2 = math.pi*D2**2/4.0
                v1 = v2 * (A2/A1)
            if v2 is None and v1 is not None and D1>0 and D2>0:
                A1 = math.pi*D1**2/4.0
                A2 = math.pi*D2**2/4.0
                v2 = v1 * (A1/A2)
            h_k = minor_loss_head(Ktot, v2 if v2 is not None else v1)
            P1 = rho * g * (Z2 - Z1 + P2/(rho*g) + (alpha*v2**2 - alpha*v1**2)/(2*g) + h_k)
            self.calc_result.setPlainText(f"P1 = {P1/1000.0:.4f} kPa")
            self.log.append("Calculadora hidráulica calculou P1.")
            return

        if P1_str and P2_str:
            P1 = float(P1_str) * 1000.0
            P2 = float(P2_str) * 1000.0
            # compute required headloss implied
            v1 = float(v1) if v1 else 0.0
            v2 = float(v2) if v2 else 0.0
            h_k = minor_loss_head(Ktot, v2 if v2!=0 else v1)
            # compute left - right of Bernoulli to check residual
            lhs = Z1 + P1/(self.rho * g) + 0.5 * v1**2 / g
            rhs = Z2 + P2/(self.rho * g) + 0.5 * v2**2 / g + h_k
            self.calc_result.setPlainText(f"LHS = {lhs:.6f} m, RHS = {rhs:.6f} m, residual = {lhs-rhs:.6e} m")
            self.log.append("Calculadora hidráulica checou consistência.")
            return

        QMessageBox.information(self, "Info", "Caso não coberto automaticamente. Forneça P1 ou P2 e pelo menos uma velocidade.")

# ----------------------------
# Dialog to add segment
# ----------------------------
class AddSegmentDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adicionar trecho")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.name = QLineEdit("Trecho")
        self.L = QLineEdit("10.0")
        self.D = QLineEdit("0.05")
        self.rug = QLineEdit("1.5e-6")
        self.elev = QLineEdit("0.0")
        self.fit1 = QLineEdit("elbow_90_short,1")  # format key,qty; semicolons múltiplos
        self.layout.addWidget(QLabel("Nome:")); self.layout.addWidget(self.name)
        self.layout.addWidget(QLabel("Comprimento L (m):")); self.layout.addWidget(self.L)
        self.layout.addWidget(QLabel("Diâmetro D (m):")); self.layout.addWidget(self.D)
        self.layout.addWidget(QLabel("Rugosidade absoluta (m):")); self.layout.addWidget(self.rug)
        self.layout.addWidget(QLabel("Δ de cota (m) (z_out - z_in):")); self.layout.addWidget(self.elev)
        self.layout.addWidget(QLabel("Fittings (formato key,qty; separar por ; ):"))
        self.layout.addWidget(self.fit1)
        btns = QHBoxLayout()
        self.layout.addLayout(btns)
        ok = QPushButton("OK"); ok.clicked.connect(self.accept); btns.addWidget(ok)
        cancel = QPushButton("Cancelar"); cancel.clicked.connect(self.reject); btns.addWidget(cancel)

    def get_segment(self):
        name = self.name.text().strip()
        L = float(self.L.text())
        D = float(self.D.text())
        rug = float(self.rug.text())
        elev = float(self.elev.text())
        fits_raw = self.fit1.text().strip()
        fits = []
        if fits_raw:
            parts = fits_raw.split(";")
            for p in parts:
                if not p.strip(): continue
                try:
                    key, qty = p.split(",")
                    fits.append((key.strip(), int(qty)))
                except:
                    continue
        return Segment(name=name, L=L, D=D, roughness=rug, fittings=fits, elevation_delta=elev)

# ----------------------------
# Main
# ----------------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
