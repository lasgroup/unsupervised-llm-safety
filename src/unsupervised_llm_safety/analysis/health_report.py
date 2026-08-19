"""Health report generator for SAE diagnostics.

Evaluates diagnostic results against configurable thresholds and
produces pass/warn/fail reports.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_THRESHOLDS = {
    "dead_neuron_pct_warn": 0.3,
    "dead_neuron_pct_fail": 0.5,
    "l0_drop_pct_warn": 0.2,
    "l0_drop_pct_fail": 0.4,
    "cluster_variance_increase_warn": 0.5,
    "cluster_variance_increase_fail": 1.0,
    "inter_cluster_distance_drop_warn": 0.5,
    "inter_cluster_distance_drop_fail": 0.75,
    "activation_concentration_warn": 0.8,
    "activation_concentration_fail": 0.95,
}


class HealthReport:
    """Evaluate SAE diagnostics against thresholds."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """Initialize with thresholds.

        Args:
            thresholds: Dict of threshold values. Missing keys use defaults.
        """
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)

    def _check(
        self,
        name: str,
        value: float,
        warn_threshold: float,
        fail_threshold: float,
        higher_is_worse: bool = True,
        description: str = "",
    ) -> Dict[str, Any]:
        """Run a single threshold check.

        Args:
            name: Check name.
            value: Measured value.
            warn_threshold: Threshold for warning.
            fail_threshold: Threshold for failure.
            higher_is_worse: If True, value > threshold is bad.
            description: Human-readable description.

        Returns:
            Dict with status, value, thresholds, message.
        """
        if higher_is_worse:
            if value >= fail_threshold:
                status = "FAIL"
            elif value >= warn_threshold:
                status = "WARN"
            else:
                status = "PASS"
        else:
            # Lower is worse (e.g., distance dropping)
            if value <= -fail_threshold:
                status = "FAIL"
            elif value <= -warn_threshold:
                status = "WARN"
            else:
                status = "PASS"

        return {
            "name": name,
            "status": status,
            "value": value,
            "warn_threshold": warn_threshold,
            "fail_threshold": fail_threshold,
            "description": description,
        }

    def evaluate(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate diagnostics results against thresholds.

        Args:
            diagnostics: Output from SAEHealthDiagnostics.run_all().

        Returns:
            Dict with per-check results and overall status.
        """
        checks = []
        t = self.thresholds
        sparsity = diagnostics.get("sparsity", {})
        cluster_var = diagnostics.get("cluster_variance", {})
        inter_dist = diagnostics.get("inter_cluster_distances", {})

        # 1. Dead neurons (post-finetune)
        dead_pct = sparsity.get("post_dead_neuron_pct", 0.0)
        checks.append(
            self._check(
                "dead_neurons",
                dead_pct,
                t["dead_neuron_pct_warn"],
                t["dead_neuron_pct_fail"],
                higher_is_worse=True,
                description=(
                    f"{dead_pct:.1%} of neurons are dead "
                    f"(never fire across all samples)"
                ),
            )
        )

        # 2. L0 collapse (drop in mean L0 from pre to post)
        pre_l0 = sparsity.get("pre_l0_mean", 0.0)
        post_l0 = sparsity.get("post_l0_mean", 0.0)
        if pre_l0 > 0:
            l0_drop_pct = (pre_l0 - post_l0) / pre_l0
        else:
            l0_drop_pct = 0.0
        checks.append(
            self._check(
                "l0_collapse",
                l0_drop_pct,
                t["l0_drop_pct_warn"],
                t["l0_drop_pct_fail"],
                higher_is_worse=True,
                description=(
                    f"Mean L0 changed from {pre_l0:.1f} to {post_l0:.1f} "
                    f"({l0_drop_pct:+.1%})"
                ),
            )
        )

        # 3. Inner-cluster variance increase
        variance_changes = cluster_var.get("variance_change_pct", {})
        max_var_increase = 0.0
        worst_cluster = -1
        for cid_str, change in variance_changes.items():
            if change > max_var_increase:
                max_var_increase = change
                worst_cluster = int(cid_str)

        checks.append(
            self._check(
                "cluster_variance_increase",
                max_var_increase,
                t["cluster_variance_increase_warn"],
                t["cluster_variance_increase_fail"],
                higher_is_worse=True,
                description=(
                    f"Max variance increase: {max_var_increase:+.1%} "
                    f"(cluster {worst_cluster})"
                ),
            )
        )

        # 4. Inter-cluster distance collapse (drop in min distance)
        min_dist_change = inter_dist.get("min_distance_change_pct", 0.0)
        checks.append(
            self._check(
                "inter_cluster_collapse",
                min_dist_change,
                t["inter_cluster_distance_drop_warn"],
                t["inter_cluster_distance_drop_fail"],
                higher_is_worse=False,  # Negative change = bad
                description=(
                    f"Min inter-cluster distance changed by "
                    f"{min_dist_change:+.1%}"
                ),
            )
        )

        # 5. Activation concentration
        post_conc = sparsity.get("post_top5_activation_frac", 0.0)
        checks.append(
            self._check(
                "activation_concentration",
                post_conc,
                t["activation_concentration_warn"],
                t["activation_concentration_fail"],
                higher_is_worse=True,
                description=(
                    f"Top 5% of neurons account for {post_conc:.1%} "
                    f"of total activation"
                ),
            )
        )

        # Overall status
        statuses = [c["status"] for c in checks]
        if "FAIL" in statuses:
            overall = "FAIL"
        elif "WARN" in statuses:
            overall = "WARN"
        else:
            overall = "PASS"

        report = {
            "overall_status": overall,
            "checks": checks,
            "n_pass": statuses.count("PASS"),
            "n_warn": statuses.count("WARN"),
            "n_fail": statuses.count("FAIL"),
        }

        return report

    def save_report(self, report: Dict[str, Any], output_dir: Path) -> None:
        """Save report as JSON and human-readable text.

        Args:
            report: Output from evaluate().
            output_dir: Directory to save reports.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON report
        with open(output_dir / "health_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Human-readable text report
        lines = []
        lines.append("=" * 60)
        lines.append("SAE HEALTH DIAGNOSTIC REPORT")
        lines.append("=" * 60)
        lines.append("")
        lines.append(
            f"Overall Status: {report['overall_status']}  "
            f"({report['n_pass']} pass, {report['n_warn']} warn, "
            f"{report['n_fail']} fail)"
        )
        lines.append("")
        lines.append("-" * 60)

        for check in report["checks"]:
            status_icon = {
                "PASS": "[PASS]",
                "WARN": "[WARN]",
                "FAIL": "[FAIL]",
            }[check["status"]]
            lines.append(f"  {status_icon} {check['name']}")
            lines.append(f"         {check['description']}")
            lines.append(
                f"         value={check['value']:.4f}  "
                f"warn={check['warn_threshold']:.4f}  "
                f"fail={check['fail_threshold']:.4f}"
            )
            lines.append("")

        lines.append("=" * 60)

        text = "\n".join(lines)
        with open(output_dir / "health_report.txt", "w") as f:
            f.write(text)

        # Also log the report
        logger.info("\n" + text)
