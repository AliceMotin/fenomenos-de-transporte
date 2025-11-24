# perdas_tubulacao_pyside6.py
# Versão B: reestruturada e convertida para PySide6
# Projeto: Estudo Computacional de Perdas de Carga em Tubulações
# GUI: PySide6, matplotlib, numpy
# Execute: python perdas_tubulacao_pyside6.py
# Referência PDF (upload local): /mnt/data/Software didatico para o ensino de Mecanica dos Fluidos e Hidraulica.pdf

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

g = 9.80665

# ----------------------------
# Hydraulic functions
# ----------------------------
def kinematic_viscosity(mu, rho):
    return mu / rho

def reynolds_number(rho, v, D, mu=None, nu=None):
    if nu is None:
        if mu is None:
            raise ValueError("Forneça mu ou nu")
        nu = kinematic_viscosity(mu, rho)
    return v * D / nu if nu != 0 else 0.0

def friction_haaland(Re, eps_rel):
    if Re <= 0:
        return float('inf')
    if Re < 2300:
        return 64.0 / Re
    A = (eps_rel / 3.7) ** 1.11
    B = 6.9 / Re
    f = (-1.8 * math.log10(A + B)) ** -2
    return f

def friction_colebrook(Re, eps_rel, tol=1e-6, maxit=200):
    if Re < 2300:
        return 64.0 / Re
    f = friction_haaland(Re, eps_rel)
    for i in range(maxit):
        lhs = 1.0 / math.sqrt(f)
        rhs = -2.0 * math.log10(eps_rel/3.7 + 2.51/(Re*math.sqrt(f)))
        resid = lhs - rhs
        if abs(resid) < tol:
            return f
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

def head_loss_darcy(L, D, v, f):
    return f * (L/D) * v**2 / (2.0 * g)

def head_loss_hazen_williams(L, D, Q, C):
    return 10.65 * L / (D ** 4.87) * (Q / C) ** 1.85

def minor_loss_head(K, v):
    return K * v**2 / (2.0 * g)

# Default tables (editable inside code)
HAZEN_WILLIAMS_C_TABLE = {
    "PVC": 150, "Aço galvanizado": 125, "Ferro fundido novo": 130, "Concreto": 120
}
DEFAULT_K_TABLE = {
    "elbow_90_short": 0.9, "elbow_90_long": 0.4, "ball_valve_open": 0.05, "pipe_entry": 0.5, "pipe_exit": 1.0
}

# ----------------------------
# Data container
# ----------------------------
class Segment:
    def __init__(self, name="Trecho", L=10.0, D=0.05, roughness=1.5e-6, fittings=None, elevation_delta=0.0):
        self.name = name
        self.L = float(L)
        self.D = float(D)
        self.roughness = float(roughness)
        self.fittings = fittings or []  # list of (key, qty)
        self.elevation_delta = float(elevation_delta)
    def total_K(self):
        s = 0.0
        for key, qty in self.fittings:
            s += DEFAULT_K_TABLE.get(key, 0.0) * qty
        return s

# ----------------------------
# Matplotlib canvas
# ----------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, dpi=100):
        fig = Figure(dpi=dpi, figsize=(6,4))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        fig.tight_layout()

# ----------------------------
# Dialog to add a segment
# ----------------------------
class AddSegmentDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adicionar trecho")
        self.setModal(True)
        layout = QFormLayout(self)

        self.name = QLineEdit("Trecho")
        self.L = QLineEdit("10.0")
        self.D = QLineEdit("0.05")
        self.rug = QLineEdit("1.5e-6")
        self.elev = QLineEdit("0.0")
        self.fit = QLineEdit("elbow_90_short,1")

        layout.addRow("Nome:", self.name)
        layout.addRow("Comprimento L (m):", self.L)
        layout.addRow("Diâmetro D (m):", self.D)
        layout.addRow("Rugosidade (m):", self.rug)
        layout.addRow("Δ cota (m):", self.elev)
        layout.addRow("Fittings (key,qty; sep ;):", self.fit)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def get_segment(self):
        name = self.name.text().strip() or "Trecho"
        try:
            L = float(self.L.text()); D = float(self.D.text()); rug = float(self.rug.text()); elev = float(self.elev.text())
        except:
            raise ValueError("Entradas numéricas inválidas no trecho")
        fits_raw = self.fit.text().strip()
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
# Main application window
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Perdas de Carga - PySide6 (versão B)")
        self.resize(1200, 780)
        self._segments = []
        self._last_results = None
        self._setup_default_pipeline()
        self._build_ui()

    def _setup_default_pipeline(self):
        s1 = Segment(name="Trecho A", L=10.0, D=0.05, roughness=1.5e-6, fittings=[("elbow_90_short",2)], elevation_delta=0.0)
        s2 = Segment(name="Trecho B", L=25.0, D=0.05, roughness=1.5e-6, fittings=[("ball_valve_open",1)], elevation_delta=2.0)
        self._segments = [s1, s2]
        self.rho = 998.2; self.mu = 1.002e-3

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        v = QVBoxLayout(central)

        tabs = QtWidgets.QTabWidget(); v.addWidget(tabs)

        tab_pipeline = QWidget(); tabs.addTab(tab_pipeline, "Pipeline"); self._build_pipeline_tab(tab_pipeline)
        tab_results = QWidget(); tabs.addTab(tab_results, "Resultados / Gráficos"); self._build_results_tab(tab_results)
        tab_cav = QWidget(); tabs.addTab(tab_cav, "Cavitação / NPSH"); self._build_cavitation_tab(tab_cav)
        tab_moody = QWidget(); tabs.addTab(tab_moody, "Diagrama de Moody"); self._build_moody_tab(tab_moody)
        tab_calc = QWidget(); tabs.addTab(tab_calc, "Calculadora Hidráulica"); self._build_calc_tab(tab_calc)

        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(140); v.addWidget(self.log)

    # ----------------------------
    # Pipeline tab
    # ----------------------------
    def _build_pipeline_tab(self, parent):
        layout = QHBoxLayout(parent)
        left = QVBoxLayout(); right = QVBoxLayout()
        layout.addLayout(left, 2); layout.addLayout(right, 3)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Nome", "L (m)", "D (m)", "Rug (m)", "Δcota (m)", "Fittings"])
        left.addWidget(self.table)
        self._refresh_segment_table()

        btns = QHBoxLayout(); left.addLayout(btns)
        add_btn = QPushButton("Adicionar trecho"); add_btn.clicked.connect(self._add_segment_dialog); btns.addWidget(add_btn)
        del_btn = QPushButton("Remover trecho"); del_btn.clicked.connect(self._remove_selected_segment); btns.addWidget(del_btn)
        load_btn = QPushButton("Carregar exemplo"); load_btn.clicked.connect(self._load_pdf_example); btns.addWidget(load_btn)

        # Right side controls
        right.addWidget(QLabel("<b>Propriedades do fluido</b>"))
        form = QHBoxLayout(); right.addLayout(form)
        lf = QVBoxLayout(); rf = QVBoxLayout(); form.addLayout(lf); form.addLayout(rf)

        lf.addWidget(QLabel("Densidade ρ (kg/m³):")); self.rho_edit = QLineEdit(str(self.rho)); lf.addWidget(self.rho_edit)
        lf.addWidget(QLabel("Viscosidade μ (Pa·s):")); self.mu_edit = QLineEdit(str(self.mu)); lf.addWidget(self.mu_edit)

        rf.addWidget(QLabel("Método f:")); self.f_method = QComboBox(); self.f_method.addItems(["Haaland","Colebrook"]); rf.addWidget(self.f_method)
        rf.addWidget(QLabel("Método perda:")); self.loss_method = QComboBox(); self.loss_method.addItems(["Darcy-Weisbach","Hazen-Williams"]); rf.addWidget(self.loss_method)

        right.addWidget(QLabel("<b>Condições</b>"))
        qlay = QHBoxLayout(); right.addLayout(qlay); qlay.addWidget(QLabel("Q (L/s):")); self.q_edit = QLineEdit("10.0"); qlay.addWidget(self.q_edit)
        play = QHBoxLayout(); right.addLayout(play); play.addWidget(QLabel("Pressão entrada (kPa):")); self.p_in_edit = QLineEdit(""); play.addWidget(self.p_in_edit)

        run_btn = QPushButton("Calcular perfil e perdas"); run_btn.clicked.connect(self._compute_pipeline); right.addWidget(run_btn)
        exp_btn = QPushButton("Exportar CSV"); exp_btn.clicked.connect(self._export_csv); right.addWidget(exp_btn)

    def _refresh_segment_table(self):
        self.table.setRowCount(len(self._segments))
        for i, s in enumerate(self._segments):
            items = [
                QTableWidgetItem(s.name), QTableWidgetItem(str(s.L)),
                QTableWidgetItem(str(s.D)), QTableWidgetItem(str(s.roughness)),
                QTableWidgetItem(str(s.elevation_delta)),
                QTableWidgetItem(", ".join([f"{k}x{q}" for k,q in s.fittings]))
            ]
            for j, it in enumerate(items):
                self.table.setItem(i, j, it)

    def _add_segment_dialog(self):
        dlg = AddSegmentDialog(self)
        if dlg.exec() == QDialog.Accepted:
            try:
                seg = dlg.get_segment()
                self._segments.append(seg)
                self._refresh_segment_table()
            except Exception as e:
                QMessageBox.warning(self, "Erro", f"Erro ao adicionar trecho: {e}")

    def _remove_selected_segment(self):
        row = self.table.currentRow()
        if row >= 0 and row < len(self._segments):
            del self._segments[row]
            self._refresh_segment_table()

    def _load_pdf_example(self):
        # keep simple: re-setup default pipeline (user can still edit)
        self._setup_default_pipeline()
        self._refresh_segment_table()
        self.log.append("Exemplo padrão carregado (base).")

    # ----------------------------
    # Results / Graph tab
    # ----------------------------
    def _build_results_tab(self, parent):
        layout = QVBoxLayout(parent)
        top = QHBoxLayout(); layout.addLayout(top)
        left = QVBoxLayout(); right = QVBoxLayout(); top.addLayout(left, 2); top.addLayout(right, 3)
        self.results_text = QTextEdit(); self.results_text.setReadOnly(True); left.addWidget(self.results_text)
        self.canvas = MplCanvas(self); right.addWidget(self.canvas)
        bl = QHBoxLayout(); layout.addLayout(bl)
        plot_btn = QPushButton("Atualizar gráfico"); plot_btn.clicked.connect(self._draw_piezometric_plot); bl.addWidget(plot_btn)
        clear_btn = QPushButton("Limpar"); clear_btn.clicked.connect(self._clear_results); bl.addWidget(clear_btn)

    def _clear_results(self):
        self.results_text.clear(); self.canvas.ax.clear(); self.canvas.draw(); self.log.append("Resultados limpos.")

    # ----------------------------
    # Cavitação / NPSH tab
    # ----------------------------
    def _build_cavitation_tab(self, parent):
        layout = QVBoxLayout(parent)
        form = QHBoxLayout(); layout.addLayout(form)
        left = QVBoxLayout(); right = QVBoxLayout(); form.addLayout(left,2); form.addLayout(right,3)

        left.addWidget(QLabel("<b>Cavitação / NPSH</b>"))
        left.addWidget(QLabel("Temperatura (°C) ou Pv (kPa):"))
        self.temp_edit = QLineEdit("20.0"); left.addWidget(self.temp_edit)
        left.addWidget(QLabel("Índice nó sucção (0=entrada):")); self.pump_node_idx = QLineEdit("0"); left.addWidget(self.pump_node_idx)
        left.addWidget(QLabel("NPSHr da bomba (m):")); self.npshr_edit = QLineEdit("3.0"); left.addWidget(self.npshr_edit)
        btn = QPushButton("Calcular NPSHa"); btn.clicked.connect(self._compute_cavitation); left.addWidget(btn)

        right.addWidget(QLabel("<b>Resultados</b>"))
        self.cav_result = QTextEdit(); self.cav_result.setReadOnly(True); right.addWidget(self.cav_result)
        self.cav_canvas = MplCanvas(self); right.addWidget(self.cav_canvas)

    # ----------------------------
    # Moody tab
    # ----------------------------
    def _build_moody_tab(self, parent):
        layout = QVBoxLayout(parent)
        top = QHBoxLayout(); layout.addLayout(top)
        left = QVBoxLayout(); right = QVBoxLayout(); top.addLayout(left,2); top.addLayout(right,3)
        left.addWidget(QLabel("<b>Diagrama de Moody</b>"))
        left.addWidget(QLabel("Re min / max:"))
        row = QHBoxLayout(); self.re_min = QLineEdit("50"); self.re_max = QLineEdit("1e7"); row.addWidget(self.re_min); row.addWidget(self.re_max); left.addLayout(row)
        left.addWidget(QLabel("eps/D (sep ,):")); self.eps_list = QLineEdit("1e-6,1e-5,5e-5"); left.addWidget(self.eps_list)
        left.addWidget(QLabel("Plotar Colebrook?")); self.colebrook_cb = QComboBox(); self.colebrook_cb.addItems(["Sim","Não"]); left.addWidget(self.colebrook_cb)
        left.addWidget(QPushButton("Atualizar Moody", clicked=self._update_moody_plot))
        self.moody_canvas = MplCanvas(self); right.addWidget(self.moody_canvas)
        layout.addWidget(QLabel("Observação: Haaland (linha sólida) e Colebrook (tracejada)."))

    # ----------------------------
    # Calculator tab (Bernoulli)
    # ----------------------------
    def _build_calc_tab(self, parent):
        layout = QVBoxLayout(parent)
        row = QHBoxLayout(); layout.addLayout(row)
        left = QVBoxLayout(); right = QVBoxLayout(); row.addLayout(left); row.addLayout(right)
        left.addWidget(QLabel("Ponto 1 - Z1 (m):")); self.Z1 = QLineEdit("10.0"); left.addWidget(self.Z1)
        left.addWidget(QLabel("P1 (kPa) - opcional:")); self.P1 = QLineEdit(""); left.addWidget(self.P1)
        left.addWidget(QLabel("D1 (m):")); self.D1 = QLineEdit("0.05"); left.addWidget(self.D1)
        left.addWidget(QLabel("v1 (m/s) - opcional:")); self.v1 = QLineEdit(""); left.addWidget(self.v1)
        right.addWidget(QLabel("Ponto 2 - Z2 (m):")); self.Z2 = QLineEdit("8.0"); right.addWidget(self.Z2)
        right.addWidget(QLabel("P2 (kPa) - opcional:")); self.P2 = QLineEdit(""); right.addWidget(self.P2)
        right.addWidget(QLabel("D2 (m):")); self.D2 = QLineEdit("0.05"); right.addWidget(self.D2)
        right.addWidget(QLabel("v2 (m/s) - opcional:")); self.v2 = QLineEdit(""); right.addWidget(self.v2)
        layout.addWidget(QLabel("K total (adimensional):")); self.K_total = QLineEdit("0.0"); layout.addWidget(self.K_total)
        bt = QPushButton("Calcular Bernoulli", clicked=self._calculate_bernoulli); layout.addWidget(bt)
        self.calc_result = QTextEdit(); self.calc_result.setReadOnly(True); layout.addWidget(self.calc_result)

    # ----------------------------
    # Compute pipeline losses
    # ----------------------------
    def _compute_pipeline(self):
        try:
            rho = float(self.rho_edit.text()); mu = float(self.mu_edit.text())
        except:
            QMessageBox.warning(self, "Erro", "Densidade ou viscosidade inválida"); return
        try:
            Q_Ls = float(self.q_edit.text())
        except:
            QMessageBox.warning(self, "Erro", "Vazão inválida"); return
        Q = Q_Ls / 1000.0
        use_hw = (self.loss_method.currentText().startswith("Hazen") or self.loss_method.currentText().startswith("Hazen"))
        use_colebrook = (self.f_method.currentText().startswith("Colebrook"))
        results = []; pressures = []; P = None
        if self.p_in_edit.text().strip() != "":
            try:
                P = float(self.p_in_edit.text()) * 1000.0
            except:
                QMessageBox.warning(self, "Erro", "Pressão de entrada inválida"); return
        P_current = P; total_h_sum = 0.0
        for seg in self._segments:
            A = math.pi * (seg.D ** 2) / 4.0; v = Q / A
            Re = reynolds_number(self.rho, v, seg.D, mu=mu); eps_rel = seg.roughness / seg.D
            if use_hw:
                C = 130.0
                try:
                    h_f = head_loss_hazen_williams(seg.L, seg.D, Q, C)
                except:
                    h_f = 0.0
                f = None
            else:
                f = friction_colebrook(Re, eps_rel) if use_colebrook else friction_haaland(Re, eps_rel)
                h_f = head_loss_darcy(seg.L, seg.D, v, f)
            Ktot = seg.total_K(); h_k = minor_loss_head(Ktot, v); total_h = h_f + h_k
            total_h_sum += total_h; delta_p = self.rho * g * total_h
            results.append({
                "segment": seg.name, "L": seg.L, "D": seg.D, "v": v, "Re": Re, "f": f,
                "h_f": h_f, "K_total": Ktot, "h_k": h_k, "h_total": total_h,
                "delta_p_Pa": delta_p, "elevation_delta": seg.elevation_delta
            })
            if P_current is not None:
                P_current = P_current - delta_p - self.rho * g * seg.elevation_delta
                pressures.append(P_current)

        # Output
        self.results_text.clear()
        if P is not None:
            self.results_text.append(f"Pressão de entrada: {P/1000.0:.3f} kPa\n")
            self.results_text.append("Pressões nos nós (kPa):")
            p_list = [P] + pressures
            for i, pval in enumerate(p_list):
                self.results_text.append(f"Nó {i}: {pval/1000.0:.3f} kPa")
        else:
            self.results_text.append(f"Perda de carga total: {total_h_sum:.6f} m (equivalente a {self.rho*g*total_h_sum:.2f} Pa)\n")

        self.results_text.append("\nDetalhes por trecho:")
        for r in results:
            line = (f"{r['segment']}: v={r['v']:.4f} m/s, Re={r['Re']:.1f}, "
                    f"f={'{:.5f}'.format(r['f']) if r['f'] else 'n/a'}, "
                    f"h_f={r['h_f']:.4f} m, Ktot={r['K_total']:.3f}, h_k={r['h_k']:.4f} m, h_total={r['h_total']:.4f} m")
            self.results_text.append(line)

        self._last_results = {"segments": results, "pressures": pressures, "P_in": P}
        self.log.append("Cálculo executado com sucesso.")
        self._draw_piezometric_plot()

    # ----------------------------
    # Plot piezometric and geometric lines
    # ----------------------------
    def _draw_piezometric_plot(self):
        if not self._last_results:
            QMessageBox.information(self, "Info", "Execute um cálculo primeiro (aba Pipeline)."); return
        res = self._last_results["segments"]; P_in = self._last_results.get("P_in", None)
        xs = [0.0]; h_piez = []; h_energy = []
        if P_in is None:
            Hp = 0.0
        else:
            Hp = P_in / (self.rho * g)
        h_piez.append(Hp); h_energy.append(Hp + 0.5 * (res[0]['v']**2) / g if res else Hp)
        for r in res:
            xs.append(xs[-1] + r['L'])
            Hp = Hp - r['h_total'] - r['elevation_delta']
            h_piez.append(Hp)
            h_energy.append(Hp + 0.5 * (r['v']**2) / g)
        # geometry (cumulative elevation)
        z_nodes = [0.0]; zc = 0.0
        for r in res:
            zc += r['elevation_delta']; z_nodes.append(zc)
        # plot
        self.canvas.ax.clear()
        self.canvas.ax.plot(xs, z_nodes, label='Tubulação (cota)', linewidth=2)
        self.canvas.ax.plot(xs, h_piez, marker='o', label='Linha piezométrica')
        self.canvas.ax.plot(xs, h_energy, marker='s', label='Linha de energia')
        self.canvas.ax.set_xlabel("Posição ao longo da tubulação (m)")
        self.canvas.ax.set_ylabel("Carga (m)")
        self.canvas.ax.grid(True)
        self.canvas.ax.legend()
        self.canvas.draw()
        self.log.append("Gráfico atualizado.")

    # ----------------------------
    # CSV export
    # ----------------------------
    def _export_csv(self):
        if not self._last_results:
            QMessageBox.information(self, "Info", "Execute um cálculo primeiro."); return
        fname, _ = QFileDialog.getSaveFileName(self, "Salvar CSV", "", "CSV Files (*.csv)")
        if not fname:
            return
        res = self._last_results["segments"]
        with open(fname, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Segment","L_m","D_m","v_m_s","Re","f","h_f_m","K_total","h_k_m","h_total_m","delta_p_Pa"])
            for r in res:
                writer.writerow([r['segment'], r['L'], r['D'], r['v'], r['Re'], r['f'], r['h_f'], r['K_total'], r['h_k'], r['h_total'], r['delta_p_Pa']])
        self.log.append(f"Exportado CSV: {fname}")

    # ----------------------------
    # Cavitation / NPSH
    # ----------------------------
    def _pv_from_temp_approx(self, T_c):
        A, B, C = 8.07131, 1730.63, 233.426
        Pv_mmHg = 10 ** (A - B / (C + T_c))
        return Pv_mmHg * 0.133322

    def _compute_cavitation(self):
        if not self._last_results:
            QMessageBox.information(self, "Info", "Execute um cálculo no Pipeline primeiro."); return
        try:
            rho = float(self.rho_edit.text())
        except:
            QMessageBox.warning(self, "Erro", "Densidade inválida"); return
        temp_or_pv = self.temp_edit.text().strip()
        # Decide: se o usuário deu um número e quer tratá-lo como Pv (kPa) ou temperatura.
        try:
            # If value is > 1000 assume kPa (unlikely) but we'll try: first try parse as float and treat as Pv
            val = float(temp_or_pv)
            # Heuristic: if between 0 and 200 -> treat as temperature (°C), else treat as Pv (kPa)
            if 0 <= val <= 200:
                Pv_kPa = self._pv_from_temp_approx(val)
            else:
                Pv_kPa = val
        except:
            Pv_kPa = 2.34
        Pv_Pa = Pv_kPa * 1000.0
        try:
            node_idx = int(self.pump_node_idx.text())
        except:
            QMessageBox.warning(self, "Erro", "Índice de nó inválido"); return
        if self._last_results.get("P_in", None) is None:
            QMessageBox.information(self, "Info", "Forneça pressão de entrada no Pipeline para cálculo de NPSHa."); return
        pressures = [self._last_results["P_in"]] + self._last_results["pressures"]
        segments = self._last_results["segments"]
        if node_idx < 0 or node_idx >= len(pressures):
            QMessageBox.warning(self, "Erro", "Índice de nó fora do intervalo."); return
        z_nodes = [0.0]; zc = 0.0
        for s in segments:
            zc += s["elevation_delta"]; z_nodes.append(zc)
        v_nodes = [seg["v"] for seg in segments]; v_nodes.append(segments[-1]["v"])
        P_node = pressures[node_idx]; z_node = z_nodes[node_idx]
        v_node = v_nodes[node_idx] if node_idx < len(v_nodes) else v_nodes[-1]
        NPSHa = (P_node / (rho * g)) + z_node - (Pv_Pa / (rho * g)) - (v_node**2) / (2.0 * g)
        p_min = min(pressures); idx_min = pressures.index(p_min); z_min = z_nodes[idx_min]
        head_min = p_min / (rho * g) + z_min; Pv_head = Pv_Pa / (rho * g)
        try:
            NPSHr = float(self.npshr_edit.text())
        except:
            NPSHr = 0.0
        txt = []
        txt.append(f"Pv usada: {Pv_kPa:.3f} kPa ({Pv_head:.4f} m)")
        txt.append(f"Nó escolhido = {node_idx}; NPSHa = {NPSHa:.4f} m; NPSHr = {NPSHr:.4f} m")
        txt.append(f"Pressão mínima: nó {idx_min} -> Pmin = {p_min/1000.0:.3f} kPa; carga = {head_min:.4f} m")
        if NPSHa < NPSHr:
            txt.append("⚠️ NPSHa < NPSHr -> risco de cavitação na bomba!")
        else:
            txt.append("✅ NPSHa >= NPSHr")
        if head_min <= Pv_head:
            txt.append("⚠️ carga mínima <= Pv -> risco de cavitação ao longo da tubulação!")
        else:
            txt.append("OK: carga mínima > Pv")
        self.cav_result.setPlainText("\n".join(txt))
        # Plot
        self.cav_canvas.ax.clear()
        xs = list(range(len(pressures)))
        heads = [Pn / (rho * g) for Pn in pressures]
        self.cav_canvas.ax.plot(xs, heads, marker='o', label='Carga piezométrica (m)')
        self.cav_canvas.ax.axhline(Pv_head, color='r', linestyle='--', label='Pv (m)')
        self.cav_canvas.ax.plot(node_idx, (P_node/(rho*g)), 'ro')
        self.cav_canvas.ax.set_xlabel("Nó (0 = entrada)")
        self.cav_canvas.ax.set_ylabel("Carga (m)")
        self.cav_canvas.ax.grid(True)
        self.cav_canvas.ax.legend()
        self.cav_canvas.draw()
        self.log.append("Cálculo cavitação realizado.")

    # ----------------------------
    # Moody diagram plot
    # ----------------------------
    def _update_moody_plot(self):
        try:
            Re_min = float(self.re_min.text()); Re_max = float(self.re_max.text())
        except:
            QMessageBox.warning(self, "Erro", "Faixa de Re inválida"); return
        try:
            eps_vals = [float(x.strip()) for x in self.eps_list.text().split(",") if x.strip()]
        except:
            QMessageBox.warning(self, "Erro", "Formato eps inválido"); return
        Re_vals = np.logspace(math.log10(max(1, Re_min)), math.log10(max(Re_min+1, Re_max)), 300)
        self.moody_canvas.ax.clear()
        for eps in eps_vals:
            f_haaland = [friction_haaland(Re, eps) if Re>0 else None for Re in Re_vals]
            self.moody_canvas.ax.loglog(Re_vals, f_haaland, label=f'Haaland eps/D={eps:.0e}', linestyle='-')
        if self.colebrook_cb.currentText().startswith("Sim"):
            for eps in eps_vals:
                f_cole = [friction_colebrook(Re, eps) if Re>0 else None for Re in Re_vals]
                self.moody_canvas.ax.loglog(Re_vals, f_cole, label=f'Colebrook eps/D={eps:.0e}', linestyle='--')
        Re_lam = np.array([1, 2300]); f_lam = 64.0 / Re_lam
        self.moody_canvas.ax.loglog(Re_lam, f_lam, 'k:', label='Laminar 64/Re')
        self.moody_canvas.ax.set_xlabel("Re")
        self.moody_canvas.ax.set_ylabel("f")
        self.moody_canvas.ax.set_title("Diagrama de Moody")
        self.moody_canvas.ax.grid(True, which='both', linestyle=':', alpha=0.5)
        self.moody_canvas.ax.legend(fontsize='small')
        self.moody_canvas.draw()
        self.log.append("Moody atualizado.")

    # ----------------------------
    # Bernoulli calculator
    # ----------------------------
    def _calculate_bernoulli(self):
        try:
            Z1 = float(self.Z1.text()); Z2 = float(self.Z2.text())
            P1_str = self.P1.text().strip(); P2_str = self.P2.text().strip()
            D1 = float(self.D1.text()); D2 = float(self.D2.text())
            v1_str = self.v1.text().strip(); v2_str = self.v2.text().strip()
            Ktot = float(self.K_total.text())
        except Exception as e:
            QMessageBox.warning(self, "Erro", "Entrada inválida: " + str(e)); return
        v1 = float(v1_str) if v1_str else None; v2 = float(v2_str) if v2_str else None
        alpha = 1.0
        if P1_str and not P2_str:
            P1 = float(P1_str) * 1000.0
            if v1 is None and v2 is None:
                QMessageBox.warning(self, "Erro", "Forneça ao menos uma velocidade"); return
            if v1 is None and v2 is not None and D1>0 and D2>0:
                A1 = math.pi*D1**2/4.0; A2 = math.pi*D2**2/4.0; v1 = v2*(A2/A1)
            if v2 is None and v1 is not None and D1>0 and D2>0:
                A1 = math.pi*D1**2/4.0; A2 = math.pi*D2**2/4.0; v2 = v1*(A1/A2)
            h_k = minor_loss_head(Ktot, v2 if v2 is not None else v1)
            P2 = self.rho * g * (Z1 - Z2 + P1/(self.rho*g) + (alpha*v1**2 - alpha*v2**2)/(2*g) - h_k)
            self.calc_result.setPlainText(f"P2 = {P2/1000.0:.4f} kPa"); self.log.append("Calculadora: P2 calculado"); return
        if P2_str and not P1_str:
            P2 = float(P2_str) * 1000.0
            if v1 is None and v2 is None:
                QMessageBox.warning(self, "Erro", "Forneça ao menos uma velocidade"); return
            if v1 is None and v2 is not None and D1>0 and D2>0:
                A1 = math.pi*D1**2/4.0; A2 = math.pi*D2**2/4.0; v1 = v2*(A2/A1)
            if v2 is None and v1 is not None and D1>0 and D2>0:
                A1 = math.pi*D1**2/4.0; A2 = math.pi*D2**2/4.0; v2 = v1*(A1/A2)
            h_k = minor_loss_head(Ktot, v2 if v2 is not None else v1)
            P1 = self.rho * g * (Z2 - Z1 + P2/(self.rho*g) + (alpha*v2**2 - alpha*v1**2)/(2*g) + h_k)
            self.calc_result.setPlainText(f"P1 = {P1/1000.0:.4f} kPa"); self.log.append("Calculadora: P1 calculado"); return
        if P1_str and P2_str:
            P1 = float(P1_str) * 1000.0; P2 = float(P2_str) * 1000.0
            v1 = float(v1) if v1 else 0.0; v2 = float(v2) if v2 else 0.0
            h_k = minor_loss_head(Ktot, v2 if v2 != 0 else v1)
            lhs = Z1 + P1/(self.rho*g) + 0.5*v1**2/g; rhs = Z2 + P2/(self.rho*g) + 0.5*v2**2/g + h_k
            self.calc_result.setPlainText(f"LHS={lhs:.6f} m, RHS={rhs:.6f} m, residual={lhs-rhs:.6e} m"); self.log.append("Calculadora: verificação feita"); return
        QMessageBox.information(self, "Info", "Caso não coberto. Forneça P1 ou P2 e pelo menos uma velocidade.")

# ----------------------------
# Main
# ----------------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
