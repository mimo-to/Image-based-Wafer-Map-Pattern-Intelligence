import yaml
import pandas as pd
import numpy as np
import os
import copy


class ROICalculator:
    def __init__(self, config_path="business/assumptions.yaml"):
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def validate_inputs(self, inputs):
        """Ensure all required business inputs are present."""
        required = [
            "factory.wafers_per_day",
            "factory.operating_days_per_month",
            "manual_inspection.time_per_wafer_minutes",
            "manual_inspection.inspector_hourly_wage_inr",
            "manual_inspection.false_negative_rate",
            "manual_inspection.false_positive_rate",
            "ai_performance.false_negative_rate",
            "ai_performance.false_positive_rate",
            "costs.wafer_cost_inr",
            "costs.defective_wafer_downstream_cost_multiplier",
            "costs.system_development_cost_inr",
            "costs.monthly_compute_cost_inr",
        ]

        missing = []
        for key in required:
            section, field = key.split(".")
            val = inputs.get(section, {}).get(field)
            if val is None:
                missing.append(key)

        if missing:
            raise ValueError(f"Missing required ROI inputs: {', '.join(missing)}")

    def calculate_labor_savings(self, inputs):
        """Calculate potential labor cost reduction."""
        f = inputs["factory"]
        m = inputs["manual_inspection"]

        # Calculate monthly hours
        wafers_month = f["wafers_per_day"] * f["operating_days_per_month"]
        hours_month = (wafers_month * m["time_per_wafer_minutes"]) / 60

        # Savings potential (Theoretical max)
        savings = hours_month * m["inspector_hourly_wage_inr"]
        return max(0, savings)

    def calculate_quality_improvement(self, inputs):
        """Calculate savings from catching missed defects."""
        f = inputs["factory"]
        m = inputs["manual_inspection"]
        ai = inputs["ai_performance"]
        c = inputs["costs"]

        wafers_month = f["wafers_per_day"] * f["operating_days_per_month"]

        # Missed defects (False Negatives)
        manual_missed = wafers_month * m["false_negative_rate"]
        ai_missed = wafers_month * ai["false_negative_rate"]

        defects_caught = max(0, manual_missed - ai_missed)

        # Cost of escaped defects
        cost_per_escape = (
            c["wafer_cost_inr"] * c["defective_wafer_downstream_cost_multiplier"]
        )
        savings = defects_caught * cost_per_escape

        return max(0, savings)

    def calculate_scrap_reduction(self, inputs):
        """Calculate savings from reducing false rejects."""
        f = inputs["factory"]
        m = inputs["manual_inspection"]
        ai = inputs["ai_performance"]
        c = inputs["costs"]

        wafers_month = f["wafers_per_day"] * f["operating_days_per_month"]

        # False Rejects (False Positives)
        manual_scrapped = wafers_month * m["false_positive_rate"]
        ai_scrapped = wafers_month * ai["false_positive_rate"]

        # Saved wafers
        wafers_saved = max(0, manual_scrapped - ai_scrapped)
        savings = wafers_saved * c["wafer_cost_inr"]

        return max(0, savings)

    def generate_report(self, user_overrides=None):
        """Generate savings report based on inputs."""
        # Merge config with user overrides (Deep Copy Fix)
        inputs = copy.deepcopy(self.config)
        if user_overrides:
            for section, data in user_overrides.items():
                if section in inputs:
                    inputs[section].update(data)

        # Validate merged inputs
        self.validate_inputs(inputs)

        # Run calculations
        labor_savings = self.calculate_labor_savings(inputs)
        quality_savings = self.calculate_quality_improvement(inputs)
        scrap_savings = self.calculate_scrap_reduction(inputs)

        total_monthly = labor_savings + quality_savings + scrap_savings

        # Payback Period
        dev_cost = inputs["costs"].get("system_development_cost_inr", 0)
        payback_months = dev_cost / total_monthly if total_monthly > 0 else float("inf")

        # Create DataFrame (Safer Wording)
        df = pd.DataFrame(
            [
                {
                    "Category": "Labor Optimization",
                    "Estimated Monthly Savings (INR)": labor_savings,
                },
                {
                    "Category": "Quality Improvement",
                    "Estimated Monthly Savings (INR)": quality_savings,
                },
                {
                    "Category": "Scrap Reduction",
                    "Estimated Monthly Savings (INR)": scrap_savings,
                },
                {"Category": "Total", "Estimated Monthly Savings (INR)": total_monthly},
            ]
        )

        # Retrieve Formulas
        formulas = inputs.get("formulas", {})

        # Format Text (Transparency Requirement)
        assumptions_text = (
            "Estimated based on provided inputs.\n\n"
            "Formulas Used:\n"
            f"- Labor: {formulas.get('labor_cost_savings_formula')}\n"
            f"- Quality: {formulas.get('quality_improvement_formula')}\n"
            f"- Scrap: {formulas.get('scrap_reduction_formula')}\n\n"
            f"- Estimated Monthly Savings: ₹{total_monthly:,.2f}\n"
            f"- Estimated Payback Period: {payback_months:.1f} months\n"
            "No guarantees implied."
        )

        return df, assumptions_text
