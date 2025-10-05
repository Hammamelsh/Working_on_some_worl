"""
STELLAR Domain Expert Validation Layer
Cleaned version with only high-value validators + Session 2 additions
"""

from scipy import stats
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ================================================================================
# CORE VALIDATORS (HIGH VALUE - KEEP)
# ================================================================================
def bradd_cost_curve_validator(results: List[Dict], df: pd.DataFrame) -> List[Dict]:
    """
    Validate cost curves using Bradd's methodology
    """
    validations = []

    for result in results:
        entity_name = result.get('Entity_Name', '')
        unit = result.get('Unit', '')
        region = result.get('Region', '')
        Y_data = result.get('Y', [])
        X_data = result.get('X', [])

        # Only validate cost curves
        if not any(indicator in unit.lower() for indicator in ['$', 'usd', 'cost', 'price', '/mwh', '/kwh']):
            continue

        valid_data = [(x, y) for x, y in zip(X_data, Y_data) if y is not None and y > 0]

        if len(valid_data) < 3:
            continue

        years, values = zip(*valid_data)

        # 1. Order-of-magnitude sanity check
        min_val, max_val = min(values), max(values)
        if max_val / min_val > 1000:  # More than 1000x difference
            validations.append({
                'check': 'Order of Magnitude',
                'product': entity_name,
                'region': region,
                'pass': False,
                'explanation': f"Extreme range detected: ${min_val:.2f} to ${max_val:.2f} (>1000x difference)",
                'evidence': {'min': min_val, 'max': max_val, 'ratio': max_val / min_val}
            })

        # 2. Wright's Law validation for technology costs
        if any(tech in entity_name.lower() for tech in ['solar', 'wind', 'battery', 'ev']):
            # Calculate year-over-year change
            yoy_changes = []
            for i in range(1, len(values)):
                yoy = (values[i] - values[i - 1]) / values[i - 1] * 100
                yoy_changes.append(yoy)

            # Technology costs should generally decline
            increasing_years = sum(1 for change in yoy_changes if change > 10)
            if increasing_years > len(yoy_changes) * 0.3:  # More than 30% of years showing increases
                validations.append({
                    'check': 'Technology Cost Trend',
                    'product': entity_name,
                    'region': region,
                    'pass': False,
                    'explanation': f"Technology cost increasing in {increasing_years}/{len(yoy_changes)} years - violates Wright's Law expectation",
                    'evidence': {'increasing_years': increasing_years, 'total_years': len(yoy_changes)}
                })
            else:
                validations.append({
                    'check': 'Technology Cost Trend',
                    'product': entity_name,
                    'region': region,
                    'pass': True,
                    'explanation': f"Cost declining as expected for technology (Wright's Law compliant)"
                })

        # 3. Context window checks (GFC, COVID)
        crisis_years = {
            2008: "Global Financial Crisis",
            2009: "GFC Recovery",
            2020: "COVID-19 Pandemic",
            2021: "COVID Recovery"
        }

        for year, crisis in crisis_years.items():
            if year in years:
                idx = years.index(year)
                if idx > 0:
                    yoy_change = (values[idx] - values[idx - 1]) / values[idx - 1] * 100
                    if abs(yoy_change) > 30:  # More than 30% change
                        validations.append({
                            'check': 'Crisis Context',
                            'product': entity_name,
                            'region': region,
                            'pass': None,  # Informational
                            'explanation': f"{crisis} ({year}): {yoy_change:+.1f}% change - expected volatility during crisis",
                            'evidence': {'year': year, 'change': yoy_change, 'context': crisis}
                        })

    return validations


def bradd_adoption_curve_validator(results: List[Dict], df: pd.DataFrame) -> List[Dict]:
    """
    Validate adoption curves using Bradd's methodology
    """
    validations = []

    for result in results:
        entity_name = result.get('Entity_Name', '')
        region = result.get('Region', '')
        metric = result.get('Metric', '')
        Y_data = result.get('Y', [])
        X_data = result.get('X', [])

        # Check if this is an adoption curve
        if not any(
                indicator in metric.lower() for indicator in ['adoption', 'sales', 'fleet', 'capacity', 'generation']):
            continue

        valid_data = [(x, y) for x, y in zip(X_data, Y_data) if y is not None and y >= 0]

        if len(valid_data) < 3:
            continue

        years, values = zip(*valid_data)

        # 1. Coverage sanity check for regional data
        if region.lower() == 'global' and any(r in str(result) for r in ['China', 'USA', 'Europe']):
            # Global should be sum of regions
            validations.append({
                'check': 'Regional Composition',
                'product': entity_name,
                'region': region,
                'pass': None,
                'explanation': "Global data present - verify sum of regional components",
                'evidence': {'type': 'composition_check'}
            })

        # 2. Fleet vs Sales lifetime check
        if 'fleet' in metric.lower() and 'sales' in str(result):
            validations.append({
                'check': 'Fleet-Sales Consistency',
                'product': entity_name,
                'region': region,
                'pass': None,
                'explanation': "Fleet should equal cumulative sales minus retirements - verify lifetime assumptions",
                'evidence': {'type': 'lifetime_check'}
            })

        # 3. S-curve expectation for new tech
        if any(tech in entity_name.lower() for tech in ['ev', 'electric', 'solar', 'wind', 'battery']):
            # Calculate growth rate
            if len(values) >= 5:
                early_growth = (values[2] - values[0]) / values[0] if values[0] > 0 else 0
                late_growth = (values[-1] - values[-3]) / values[-3] if values[-3] > 0 else 0

                # New tech should show accelerating then decelerating growth (S-curve)
                if early_growth > 0 and late_growth > 0:
                    if late_growth > early_growth * 2:  # Still accelerating
                        validations.append({
                            'check': 'S-Curve Pattern',
                            'product': entity_name,
                            'region': region,
                            'pass': True,
                            'explanation': "Technology showing expected S-curve adoption pattern",
                            'evidence': {'early_growth': early_growth, 'late_growth': late_growth}
                        })

        # 4. Order of magnitude by region
        region_expectations = {
            'china': {'min_scale': 0.2, 'max_scale': 0.5},  # China typically 20-50% of global
            'usa': {'min_scale': 0.1, 'max_scale': 0.3},  # USA typically 10-30% of global
            'europe': {'min_scale': 0.1, 'max_scale': 0.25},  # Europe typically 10-25% of global
        }

        if region.lower() in region_expectations and max(values) > 0:
            expected = region_expectations[region.lower()]
            validations.append({
                'check': 'Regional Scale',
                'product': entity_name,
                'region': region,
                'pass': None,
                'explanation': f"{region} typically represents {expected['min_scale'] * 100:.0f}-{expected['max_scale'] * 100:.0f}% of global market",
                'evidence': {'expected_range': expected}
            })

    return validations


def inflation_adjustment_validator(results: List[Dict], df: pd.DataFrame) -> List[Dict]:
    """
    Check if cost data needs inflation adjustment (Bradd's rule #4)
    """
    validations = []

    # Approximate inflation rates (simplified)
    cumulative_inflation = {
        2010: 1.00, 2011: 1.03, 2012: 1.05, 2013: 1.07, 2014: 1.08,
        2015: 1.09, 2016: 1.10, 2017: 1.12, 2018: 1.15, 2019: 1.17,
        2020: 1.18, 2021: 1.24, 2022: 1.34, 2023: 1.39, 2024: 1.44, 2025: 1.47
    }

    for result in results:
        entity_name = result.get('Entity_Name', '')
        unit = result.get('Unit', '')

        # Only check cost/price data
        if not any(indicator in unit.lower() for indicator in ['$', 'usd', 'cost', 'price']):
            continue

        Y_data = result.get('Y', [])
        X_data = result.get('X', [])

        valid_data = [(x, y) for x, y in zip(X_data, Y_data) if y is not None and y > 0]

        if len(valid_data) < 5:
            continue

        years, values = zip(*valid_data)

        # Check if values appear to be nominal (not inflation adjusted)
        first_year = min(years)
        last_year = max(years)

        if first_year in cumulative_inflation and last_year in cumulative_inflation:
            expected_inflation = cumulative_inflation[last_year] / cumulative_inflation[first_year]
            actual_change = values[-1] / values[0]

            # If costs increased roughly in line with inflation, likely nominal
            if 0.8 < actual_change / expected_inflation < 1.3:
                validations.append({
                    'check': 'Inflation Adjustment',
                    'product': entity_name,
                    'region': result.get('Region', ''),
                    'pass': False,
                    'explanation': f"Cost data appears to be in nominal terms - recommend converting to real (inflation-adjusted) values",
                    'evidence': {
                        'expected_inflation': f"{(expected_inflation - 1) * 100:.1f}%",
                        'actual_change': f"{(actual_change - 1) * 100:.1f}%"
                    }
                })

    return validations
def validate_units_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing or inconsistent units and scaling in the data.
    Returns a DataFrame of issues (Product, Region, Metric, Unit, Issue description).
    """
    issues = []
    scale_terms = ["thousand", "million", "billion"]
    for idx, row in df.iterrows():
        unit = str(row.get('unit', '') or '')
        metric = str(row.get('metric', '') or '').lower()
        product = row.get('product_name', '')
        region = row.get('region', '')
        issue = None
        # Missing unit or unit only a scale word
        if unit.strip() == "" or unit.lower() in ["nan", "none"]:
            issue = "Missing unit"
        elif any(unit.lower() == term or unit.lower() == term + "s" for term in scale_terms):
            issue = "Unit only provides scale, not measure"
        # Mismatched unit vs metric context
        if issue is None:
            # Cost metric but no currency in unit
            if any(word in metric for word in ["cost", "price", "lcoe"]) and ("$" not in unit and "usd" not in unit.lower()):
                issue = "Cost metric without currency unit"
            # Capacity/energy metric but unit has currency
            if any(word in metric for word in ["capacity", "generation", "production", "consumption"]) and any(cur in unit.lower() for cur in ["$", "usd"]):
                issue = "Non-cost metric with currency unit"
            # Power vs energy unit mismatches
            if unit.lower().endswith("wh") and any(word in metric for word in ["capacity", "installed"]):
                issue = "Energy unit used for capacity metric"
            if (unit.lower().endswith("w") and not unit.lower().endswith("wh")) and any(word in metric for word in ["generation", "consumption"]):
                issue = "Power unit used for energy metric"
        if issue:
            issues.append({
                "Product": product,
                "Region": region,
                "Metric": row.get('metric', ''),
                "Unit": unit,
                "Issue": issue
            })
    issues_df = pd.DataFrame(issues).drop_duplicates()
    return issues_df

def check_year_anomalies(results: List[Dict[str, Any]], df: pd.DataFrame) -> List[str]:
    """
    Detect anomalous or out-of-order years in the dataset.
    Returns a list of warning messages for any year issues found.
    """
    issues = []
    # Check for invalid year values (0, very large years like 9999)
    if not df.empty and 'year' in df.columns:
        anomaly_years = df[(df['year'] <= 0) | (df['year'] >= 9999)]
        for _, row in anomaly_years.iterrows():
            year_val = int(row['year'])
            issues.append(f"{row['product_name']} — {row['region']}: Contains anomalous year {year_val}")
    # Check for any decreasing year sequences in original results
    for res in results:
        years = res.get('X', [])
        if not years or not all(isinstance(y, (int, float)) for y in years):
            continue
        sorted_years = sorted([y for y in years if isinstance(y, (int, float))])
        if sorted_years != [y for y in years if isinstance(y, (int, float))]:
            name = res.get('Entity_Name', 'Unknown')
            region = res.get('Region', 'Unknown')
            issues.append(f"{name} — {region}: Year sequence is not sorted chronologically")
    # Deduplicate any repeated messages
    issues = list(dict.fromkeys(issues))
    return issues

def cost_parity_threshold(df: pd.DataFrame, parity_level: float = 70, cutoff_year: int = 2022) -> List[Dict[str, Any]]:
    """
    Identify whether solar/wind cost curves reach conventional cost parity by expected years.
    Critical for Tony Seba framework - cost parity is THE disruption trigger.
    """
    outputs = []
    if df.empty:
        return outputs

    for (product, region), group in df.groupby(["product_name", "region"]):
        name = str(product).lower()
        unit = str(group["unit"].iloc[0]).lower() if not group.empty else ""
        metric = str(group["metric"].iloc[0]).lower() if not group.empty else ""

        if ("solar" in name or "wind" in name) and ("/mwh" in unit or "/kwh" in unit or "lcoe" in metric):
            subset = group.copy()
            subset["year"] = pd.to_numeric(subset["year"], errors="coerce")
            recent = subset[subset["year"] <= cutoff_year]
            if recent.empty:
                recent = subset

            costs = recent["value"].astype(float).tolist()
            years = recent["year"].astype(int).tolist()

            passed = True
            fail_year = None

            if costs:
                if all(val > parity_level for val in costs):
                    passed = False
                    fail_year = max(years) if years else None
                else:
                    for y, val in sorted(zip(years, costs)):
                        if val <= parity_level:
                            fail_year = y
                            break
                    if fail_year and fail_year > cutoff_year:
                        passed = False

            explanation = "Costs reach parity on time" if passed else (
                f"Costs stay above ${parity_level:.0f}/MWh through {fail_year}" if fail_year else
                f"Costs never drop below ${parity_level:.0f}/MWh")

            outputs.append({
                "product": product,
                "region": region,
                "pass": passed,
                "check": "Cost Parity Threshold",
                "explanation": explanation
            })
    return outputs

def adoption_saturation_feasibility(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Check if projected adoption levels exceed plausible saturation (95%).
    Simple reality check on overly optimistic projections.
    """
    outputs = []
    ev_shares = {}
    res_map = {(res.get('Entity_Name'), res.get('Region')): res for res in results}

    for (entity, region), res in list(res_map.items()):
        if entity and "(EV)" in entity:
            base_name = entity.replace("(EV)", "").strip()
            base_res = res_map.get((base_name, region))
            if base_res:
                base_unit = str(base_res.get('Unit', '')).lower()
                ev_unit = str(res.get('Unit', '')).lower()
                scale_base = 1000.0 if "thousand" in base_unit else (1000000.0 if "million" in base_unit else 1.0)
                scale_ev = 1000.0 if "thousand" in ev_unit else (1000000.0 if "million" in ev_unit else 1.0)
                max_share = 0.0

                for year, ev_val in zip(res.get('X', []), res.get('Y', [])):
                    if not isinstance(year, (int, float)):
                        continue
                    try:
                        base_index = base_res['X'].index(year)
                    except ValueError:
                        continue
                    base_val = base_res['Y'][base_index]
                    if base_val and ev_val:
                        share = (ev_val * scale_ev) / (base_val * scale_base) if base_val * scale_base > 0 else 0
                        if share > max_share:
                            max_share = share
                ev_shares[(entity, region)] = max_share

    for (entity, region), share in ev_shares.items():
        percent = share * 100
        passed = share <= 0.95
        status = "OK" if passed else "UNLIKELY"
        outputs.append({
            "product": entity,
            "region": region,
            "pass": passed,
            "check": "Adoption Saturation Feasibility",
            "explanation": f"{status} ({percent:.0f}%)"
        })
    return outputs


def oil_displacement_check(results: List[Dict[str, Any]],
                           df: pd.DataFrame,
                           material_pct: float = 0.10,
                           high_adoption: float = 0.70,
                           regional_params: Optional[Dict[str, Dict]] = None) -> List[Dict[str, Any]]:
    """
    Calculate expected oil displacement from EV adoption using physics-based formula.
    Critical for validating energy transition credibility.
    CORRECTED: Proper daily displacement calculation and display
    """
    outputs = []

    # Default regional parameters if not provided
    default_params = {
        'global': {'km_per_vehicle_per_year': 15000, 'liters_per_100km': 8.0},
        'usa': {'km_per_vehicle_per_year': 18000, 'liters_per_100km': 10.0},
        'china': {'km_per_vehicle_per_year': 12000, 'liters_per_100km': 7.0},
        'europe': {'km_per_vehicle_per_year': 13000, 'liters_per_100km': 6.5},
    }

    if regional_params is None:
        regional_params = default_params

    res_map = {(res.get('Entity_Name'), res.get('Region')): res for res in results}

    for (entity, region), res in list(res_map.items()):
        if not entity or "(EV)" not in entity:
            continue

        base_name = entity.replace("(EV)", "").strip()
        base_res = res_map.get((base_name, region))

        if not base_res:
            continue

        region_key = region.lower()
        params = regional_params.get(region_key, regional_params.get('global', default_params['global']))
        km_per_year = params['km_per_vehicle_per_year']
        liters_per_100km = params['liters_per_100km']

        latest_displacement = None
        latest_year = None

        for year, ev_val in zip(res.get('X', []), res.get('Y', [])):
            if ev_val is None or not isinstance(year, (int, float)):
                continue

            # Handle unit scaling properly
            ev_unit = str(res.get('Unit', '')).lower()

            # Determine scale multiplier
            scale_ev = 1.0
            if "thousand" in ev_unit:
                scale_ev = 1000.0
            elif "million" in ev_unit:
                scale_ev = 1000000.0
            elif "billion" in ev_unit:
                scale_ev = 1000000000.0

            # Get actual EV fleet size
            ev_fleet = ev_val * scale_ev

            # Calculate annual fuel consumption displaced
            annual_km_displaced = ev_fleet * km_per_year
            annual_liters_displaced = annual_km_displaced * (liters_per_100km / 100)

            # Convert to barrels (159 liters = 1 barrel)
            annual_barrels_displaced = annual_liters_displaced / 159.0

            # CORRECTED: For daily displacement, divide annual by 365
            # World oil consumption is ~100 million barrels/day for context
            daily_barrels_displaced = annual_barrels_displaced / 365.0

            latest_displacement = daily_barrels_displaced
            latest_year = year

        if latest_displacement is not None:
            # Look for reported displacement data
            displacement_found = False
            for check_res in results:
                if (check_res.get('Region') == region and
                        "displacement" in str(check_res.get('Metric', '')).lower()):
                    displacement_found = True
                    if check_res.get('Y'):
                        reported = check_res['Y'][-1]
                        if reported:
                            # Handle unit scaling in reported data
                            reported_unit = str(check_res.get('Unit', '')).lower()

                            # Convert reported value to barrels/day
                            if "thousand" in reported_unit or "kb/d" in reported_unit:
                                reported = reported * 1000
                            elif "million" in reported_unit or "mb/d" in reported_unit:
                                reported = reported * 1000000
                            elif "bbl/year" in reported_unit or "barrels/year" in reported_unit:
                                reported = reported / 365.0  # Convert annual to daily

                            # Calculate difference
                            diff_pct = abs(latest_displacement - reported) / reported * 100 if reported > 0 else 999

                            # Allow 50% tolerance for this complex calculation
                            passed = diff_pct <= 50

                            # CORRECTED: Display in thousands of barrels per day for readability
                            explanation = (f"Derived: {latest_displacement / 1000:.1f} kb/d, "
                                           f"Reported: {reported / 1000:.1f} kb/d ({diff_pct:.0f}% diff)")
                        else:
                            passed = False
                            explanation = f"Derived {latest_displacement / 1000:.1f} kb/d but reported value missing"
                    break

            if not displacement_found:
                passed = False
                explanation = f"Warning: Derived {latest_displacement / 1000:.1f} kb/d but no displacement series found"

            # Reality check: Global oil displacement shouldn't exceed 10 million barrels/day by 2025
            if latest_displacement > 10000000:
                passed = False
                explanation = f"ERROR: Unrealistic displacement ({latest_displacement / 1000000:.1f} mb/d exceeds global feasibility)"
            elif latest_displacement > 1000000:  # More than 1 million barrels/day needs scrutiny
                if passed:  # Only add warning if otherwise passing
                    explanation += " - High value needs verification"

            outputs.append({
                "product": entity,
                "region": region,
                "pass": passed if displacement_found else False,
                "check": "Oil Displacement",
                "explanation": explanation,
                "evidence": {
                    "derived_displacement_bbl_per_day": latest_displacement,
                    "derived_displacement_kb_per_day": latest_displacement / 1000,
                    "year": latest_year,
                    "ev_fleet_size": ev_fleet,
                    "km_per_year": km_per_year,
                    "liters_per_100km": liters_per_100km
                }
            })

    return outputs
# ================================================================================
# HIGH VALUE VALIDATORS FROM SESSION 1
# ================================================================================

def derived_oil_displacement_validator(results: List[Dict[str, Any]],
                                      regional_params: Optional[Dict[str, Dict]] = None) -> List[Dict[str, Any]]:
    """
    Calculate expected oil displacement from EV adoption using physics-based formula.
    Critical for validating energy transition credibility.
    """
    outputs = []

    # Default regional parameters if not provided
    default_params = {
        'global': {'km_per_vehicle_per_year': 15000, 'liters_per_100km': 8.0},
        'usa': {'km_per_vehicle_per_year': 18000, 'liters_per_100km': 10.0},
        'china': {'km_per_vehicle_per_year': 12000, 'liters_per_100km': 7.0},
        'europe': {'km_per_vehicle_per_year': 13000, 'liters_per_100km': 6.5},
    }

    if regional_params is None:
        regional_params = default_params

    res_map = {(res.get('Entity_Name'), res.get('Region')): res for res in results}

    for (entity, region), res in list(res_map.items()):
        if not entity or "(EV)" not in entity:
            continue

        base_name = entity.replace("(EV)", "").strip()
        base_res = res_map.get((base_name, region))

        if not base_res:
            continue

        region_key = region.lower()
        params = regional_params.get(region_key, regional_params.get('global', default_params['global']))
        km_per_year = params['km_per_vehicle_per_year']
        liters_per_100km = params['liters_per_100km']

        latest_displacement = None
        latest_year = None

        for year, ev_val in zip(res.get('X', []), res.get('Y', [])):
            if ev_val is None or not isinstance(year, (int, float)):
                continue

            ev_unit = str(res.get('Unit', '')).lower()
            scale_ev = 1000.0 if "thousand" in ev_unit else (1000000.0 if "million" in ev_unit else 1.0)
            ev_fleet = ev_val * scale_ev

            liters_saved = ev_fleet * km_per_year * (liters_per_100km / 100)
            barrels_saved = liters_saved / 159.0

            latest_displacement = barrels_saved
            latest_year = year

        if latest_displacement is not None:
            displacement_found = False
            for check_res in results:
                if (check_res.get('Region') == region and
                    "displacement" in str(check_res.get('Metric', '')).lower()):
                    displacement_found = True
                    if check_res.get('Y'):
                        reported = check_res['Y'][-1]
                        if reported:
                            diff_pct = abs(latest_displacement - reported) / reported * 100
                            passed = diff_pct <= 20
                            explanation = (f"Derived: {latest_displacement:.0f} bbl/day, "
                                         f"Reported: {reported:.0f} bbl/day ({diff_pct:.0f}% diff)")
                        else:
                            passed = False
                            explanation = f"Derived {latest_displacement:.0f} bbl/day but reported value missing"
                    break

            if not displacement_found:
                passed = False
                explanation = f"Warning: Derived {latest_displacement:.0f} bbl/day but no displacement series found"

            outputs.append({
                "product": entity,
                "region": region,
                "pass": passed if displacement_found else False,
                "check": "Derived Oil Displacement",
                "explanation": explanation,
                "evidence": {
                    "derived_displacement": latest_displacement,
                    "year": latest_year,
                    "km_per_year": km_per_year,
                    "liters_per_100km": liters_per_100km
                }
            })

    return outputs

def global_oil_sanity_validator(df: pd.DataFrame,
                               transport_band: Tuple[float, float] = (55, 60),
                               total_band: Tuple[float, float] = (90, 110),
                               crisis_windows: List[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
    """
    Check global oil demand stays within realistic bounds (~100 mb/d ±10%).
    Critical ground-truth check against well-known benchmarks.
    """
    outputs = []

    if crisis_windows is None:
        crisis_windows = [(2020, 2021)]  # COVID period

    if df.empty:
        return outputs

    global_oil = df[df['region'].str.lower() == 'global']

    for _, row in global_oil.iterrows():
        metric = str(row['metric']).lower()
        year = row['year']
        value = row['value']

        in_crisis = any(start <= year <= end for start, end in crisis_windows)

        if "transport" in metric and "oil" in metric:
            if not in_crisis and not (transport_band[0] <= value <= transport_band[1]):
                outputs.append({
                    "product": "Global Oil",
                    "region": "Global",
                    "pass": False,
                    "check": "Global Oil Sanity",
                    "explanation": f"Transport oil {value:.1f} mb/d outside expected {transport_band} mb/d in {year}"
                })

        elif "total" in metric and "oil" in metric:
            if not in_crisis and not (total_band[0] <= value <= total_band[1]):
                outputs.append({
                    "product": "Global Oil",
                    "region": "Global",
                    "pass": False,
                    "check": "Global Oil Sanity",
                    "explanation": f"Total oil {value:.1f} mb/d outside expected {total_band} mb/d in {year}"
                })

    if not outputs and not global_oil.empty:
        outputs.append({
            "product": "Global Oil",
            "region": "Global",
            "pass": True,
            "check": "Global Oil Sanity",
            "explanation": "Global oil demand within expected bands"
        })

    return outputs

def capacity_factor_validator(capacity_df: pd.DataFrame,
                             generation_df: pd.DataFrame,
                             decline_years: int = 3) -> List[Dict[str, Any]]:
    """
    Detect declining capacity factors - THE leading indicator of stranded assets.
    Critical for Tony Seba framework disruption detection.
    """
    outputs = []

    if capacity_df.empty or generation_df.empty:
        return outputs

    merged = pd.merge(
        capacity_df[['product_name', 'region', 'year', 'value']],
        generation_df[['product_name', 'region', 'year', 'value']],
        on=['product_name', 'region', 'year'],
        suffixes=('_capacity', '_generation')
    )

    if merged.empty:
        return outputs

    for (product, region), group in merged.groupby(['product_name', 'region']):
        group = group.sort_values('year')

        hours_per_year = 8760
        group['capacity_factor'] = group['value_generation'] / (group['value_capacity'] * hours_per_year)

        cf_values = group['capacity_factor'].values
        years = group['year'].values

        if len(cf_values) >= decline_years:
            declining_streak = 0
            for i in range(1, len(cf_values)):
                if cf_values[i] < cf_values[i-1]:
                    declining_streak += 1
                    if declining_streak >= decline_years - 1:
                        outputs.append({
                            "product": product,
                            "region": region,
                            "pass": False,
                            "check": "Capacity Factor Trend",
                            "explanation": f"Capacity factor declining for {declining_streak+1} consecutive years - stranded asset risk",
                            "evidence": {
                                "years": years.tolist(),
                                "capacity_factors": cf_values.tolist()
                            }
                        })
                        break
                else:
                    declining_streak = 0

            if declining_streak < decline_years - 1:
                outputs.append({
                    "product": product,
                    "region": region,
                    "pass": True,
                    "check": "Capacity Factor Trend",
                    "explanation": "No persistent capacity factor decline detected"
                })

    return outputs

# ================================================================================
# NEW VALIDATORS FROM SESSION 2 (CRITICAL ADDITIONS)
# ================================================================================

def market_context_validator(df: pd.DataFrame, results: List[Dict[str, Any]],
                             llm_client=None) -> List[Dict[str, Any]]:
    """
    Links data anomalies to historical events with technology lifecycle awareness.
    Shows explanations for why large changes might be acceptable.
    """
    outputs = []

    # Known major events that explain large changes
    known_events = {
        1973: "Oil Crisis - OPEC embargo",
        1979: "Iranian Revolution - oil supply disruption",
        1991: "USSR collapse - Russia data shifts to Europe",
        2008: "Global Financial Crisis",
        2011: "Fukushima disaster - nuclear shutdowns",
        2014: "Oil price crash - shale revolution",
        2020: "COVID-19 pandemic - demand destruction",
        2021: "COVID recovery - demand rebound",
        2022: "Russia-Ukraine conflict - energy crisis"
    }

    # Technology lifecycle thresholds
    def get_lifecycle_threshold(product: str, year: int) -> float:
        product_lower = product.lower()

        if any(tech in product_lower for tech in ['hydrogen', 'carbon capture']):
            return 3.0  # 300% acceptable for emerging tech

        if 'solar' in product_lower or 'battery' in product_lower:
            if year < 2010:
                return 3.0  # Early phase
            elif year < 2020:
                return 2.0  # Growth phase
            else:
                return 1.0  # Mainstream

        if any(tech in product_lower for tech in ['coal', 'nuclear']):
            return 0.5  # Mature tech

        return 1.0  # Default

    # Process each product/region
    for (product, region), group in df.groupby(['product_name', 'region']):
        group = group.sort_values('year')
        values = group['value'].values
        years = group['year'].values

        for i in range(1, len(values)):
            if values[i - 1] > 0:
                yoy_change = (values[i] - values[i - 1]) / values[i - 1]
                year = int(years[i])

                # Only flag if change exceeds 30%
                if abs(yoy_change) > 0.30:
                    # Check for known events
                    event_context = None
                    for event_year, event_desc in known_events.items():
                        if abs(year - event_year) <= 1:
                            event_context = event_desc
                            break

                    # Check data source change
                    source_change = False
                    for res in results:
                        if res.get('Entity_Name') == product and res.get('Region') == region:
                            urls = res.get('DataSource_URLs', [])
                            if len(set(urls)) > 1:
                                source_change = True
                                break

                    # Determine pass/fail with explanation
                    lifecycle_threshold = get_lifecycle_threshold(product, year)

                    if event_context:
                        explanation = f"{yoy_change * 100:.0f}% change explained by: {event_context}"
                        passed = True
                    elif source_change:
                        explanation = f"{yoy_change * 100:.0f}% change possibly due to data source change"
                        passed = None  # Uncertain
                    elif abs(yoy_change) <= lifecycle_threshold:
                        explanation = f"{yoy_change * 100:.0f}% change within lifecycle expectations for {product}"
                        passed = True
                    else:
                        explanation = f"{yoy_change * 100:.0f}% unexplained change - needs investigation"
                        passed = False

                    outputs.append({
                        "product": product,
                        "region": region,
                        "pass": passed,
                        "check": "Market Context",
                        "explanation": explanation,
                        "evidence": {
                            "year": year,
                            "change_percent": yoy_change * 100,
                            "event": event_context,
                            "source_change": source_change
                        }
                    })

    return outputs


def global_sum_validator(df: pd.DataFrame, tolerance: float = 0.05) -> List[Dict[str, Any]]:
    """
    Verify that global values equal the sum of regional values.
    Critical for data consistency and preventing double-counting.
    """
    outputs = []

    if df.empty:
        return outputs

    # Group by product, metric, and year
    for (product, metric, year), group in df.groupby(['product_name', 'metric', 'year']):
        regions = group['region'].str.lower().tolist()

        # Check if we have both global and regional data
        if 'global' in regions and len(regions) > 1:
            global_value = group[group['region'].str.lower() == 'global']['value'].iloc[0]

            # Define expected regional components
            main_regions = ['usa', 'china', 'europe', 'india', 'japan']
            regional_sum = 0
            regions_found = []

            for region in main_regions:
                region_data = group[group['region'].str.lower() == region]
                if not region_data.empty:
                    regional_sum += region_data['value'].iloc[0]
                    regions_found.append(region)

            # Add "rest of world" or "other" if present
            for rest_term in ['rest of world', 'row', 'other', 'others']:
                rest_data = group[group['region'].str.lower().str.contains(rest_term, na=False)]
                if not rest_data.empty:
                    regional_sum += rest_data['value'].iloc[0]
                    regions_found.append(rest_term)

            # Check if sum matches global
            if regional_sum > 0:
                diff_pct = abs(global_value - regional_sum) / global_value if global_value > 0 else 1.0

                if diff_pct <= tolerance:
                    outputs.append({
                        "product": product,
                        "region": "Global",
                        "pass": True,
                        "check": "Global Sum Validation",
                        "explanation": f"Global value matches sum of regions (within {tolerance * 100:.0f}% tolerance)",
                        "evidence": {
                            "global": global_value,
                            "regional_sum": regional_sum,
                            "regions_included": regions_found
                        }
                    })
                else:
                    outputs.append({
                        "product": product,
                        "region": "Global",
                        "pass": False,
                        "check": "Global Sum Validation",
                        "explanation": f"Global ({global_value:.1f}) != sum of regions ({regional_sum:.1f}) - {diff_pct * 100:.1f}% difference",
                        "evidence": {
                            "global": global_value,
                            "regional_sum": regional_sum,
                            "difference_pct": diff_pct * 100,
                            "regions_included": regions_found
                        }
                    })

    return outputs


def data_source_quality_validator(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate that data comes from reputable sources recognized by domain experts.
    """
    outputs = []

    # Tier 1: Gold standard sources
    tier1_sources = [
        'iea.org',  # International Energy Agency
        'eia.gov',  # US Energy Information Administration
        'bp.com/statistical',  # BP Statistical Review
        'bnef.com',  # Bloomberg New Energy Finance
        'irena.org',  # International Renewable Energy Agency
        'usgs.gov',  # US Geological Survey
        'worldbank.org',  # World Bank
        'imf.org',  # International Monetary Fund
        'un.org',  # United Nations
        'opec.org'  # OPEC
    ]

    # Tier 2: Good secondary sources
    tier2_sources = [
        'statista.com',
        'ourworldindata.org',
        'ember-climate.org',
        'rystadenergy.com',
        'woodmac.com',
        'mckinsey.com',
        'ieefa.org',
        'seia.org',
        'gwec.net'
    ]

    # Tier 3: Caution needed
    tier3_sources = [
        'wikipedia.org',
        'tradingeconomics.com',
        'indexmundi.com'
    ]

    for result in results:
        entity_name = result.get('Entity_Name', 'Unknown')
        region = result.get('Region', 'Unknown')
        urls = result.get('DataSource_URLs', [])

        if not urls:
            outputs.append({
                "product": entity_name,
                "region": region,
                "pass": False,
                "check": "Data Source Quality",
                "explanation": "No data source provided",
                "severity": "critical"
            })
            continue

        # Analyze source quality
        source_tiers = []
        for url in urls:
            url_lower = str(url).lower()

            if any(t1 in url_lower for t1 in tier1_sources):
                source_tiers.append(1)
            elif any(t2 in url_lower for t2 in tier2_sources):
                source_tiers.append(2)
            elif any(t3 in url_lower for t3 in tier3_sources):
                source_tiers.append(3)
            else:
                source_tiers.append(4)  # Unknown source

        # Determine overall quality
        best_tier = min(source_tiers) if source_tiers else 4

        if best_tier == 1:
            outputs.append({
                "product": entity_name,
                "region": region,
                "pass": True,
                "check": "Data Source Quality",
                "explanation": "Gold standard source (Tier 1)",
                "evidence": {"sources": urls, "tier": 1}
            })
        elif best_tier == 2:
            outputs.append({
                "product": entity_name,
                "region": region,
                "pass": True,
                "check": "Data Source Quality",
                "explanation": "Reputable secondary source (Tier 2)",
                "evidence": {"sources": urls, "tier": 2}
            })
        elif best_tier == 3:
            outputs.append({
                "product": entity_name,
                "region": region,
                "pass": None,
                "check": "Data Source Quality",
                "explanation": "Source requires verification (Tier 3)",
                "evidence": {"sources": urls, "tier": 3}
            })
        else:
            outputs.append({
                "product": entity_name,
                "region": region,
                "pass": False,
                "check": "Data Source Quality",
                "explanation": "Unknown/unverified source - needs domain expert review",
                "evidence": {"sources": urls, "tier": 4}
            })

    return outputs

def _get_llm_context(llm_client, product: str, region: str, year: int, change: float) -> str:
    """Helper to get LLM explanation for data anomaly"""
    try:
        prompt = f"""
        Explain this energy data anomaly in one sentence:
        - Product: {product}
        - Region: {region}
        - Year: {year}
        - Change: {change*100:.0f}%
        
        What major world event or market condition likely caused this?
        """

        response = llm_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )

        return response.choices[0].message.content.strip()
    except:
        return f"{change*100:.0f}% change - LLM context unavailable"

def data_source_integrity_validator(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flags when data sources change mid-series.
    Critical for transparency about potential artificial jumps/drops.
    """
    outputs = []

    for result in results:
        entity_name = result.get('Entity_Name', 'Unknown')
        region = result.get('Region', 'Unknown')
        urls = result.get('DataSource_URLs', [])
        years = result.get('X', [])

        if len(set(urls)) > 1 and years:
            # Multiple sources detected
            # Try to identify where source changes (simplified - would need more metadata)
            outputs.append({
                "product": entity_name,
                "region": region,
                "pass": None,  # Info only
                "check": "Data Source Integrity",
                "explanation": f"Multiple sources detected ({len(set(urls))} different) - potential discontinuity",
                "severity": "info",
                "recommendation": "Add visual marker at source transition points"
            })

    return outputs

def metric_validity_validator(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    CRITICAL: Warns when metrics are inappropriate (e.g., LCOE for declining oil systems).
    Implements Adam's critique about misleading LCOE for declining systems.
    """
    outputs = []

    if df.empty:
        return outputs

    for (product, region), group in df.groupby(['product_name', 'region']):
        product_lower = product.lower()

        # Check for LCOE on oil/gas
        if ('oil' in product_lower or 'gas' in product_lower or 'coal' in product_lower):
            lcoe_data = group[
                (group['metric'].str.contains('lcoe', case=False, na=False)) |
                (group['unit'].str.contains('/mwh', case=False, na=False)) |
                (group['unit'].str.contains('/kwh', case=False, na=False))
            ]

            if not lcoe_data.empty:
                # Check if this is a declining system
                capacity = group[group['metric'].str.contains('capacity', case=False, na=False)]
                generation = group[group['metric'].str.contains('generation', case=False, na=False)]

                is_declining = False
                explanation_detail = ""

                if not capacity.empty and not generation.empty:
                    cap_by_year = capacity.groupby('year')['value'].mean()
                    gen_by_year = generation.groupby('year')['value'].mean()

                    if len(cap_by_year) >= 3 and len(gen_by_year) >= 3:
                        # Check trends
                        cap_trend = (cap_by_year.iloc[-1] - cap_by_year.iloc[0]) / cap_by_year.iloc[0] if cap_by_year.iloc[0] > 0 else 0
                        gen_trend = (gen_by_year.iloc[-1] - gen_by_year.iloc[0]) / gen_by_year.iloc[0] if gen_by_year.iloc[0] > 0 else 0

                        if cap_trend > 0.1 and gen_trend < -0.1:  # Capacity up >10%, generation down >10%
                            is_declining = True
                            explanation_detail = f"Capacity +{cap_trend*100:.0f}% but generation {gen_trend*100:.0f}%"

                if is_declining:
                    outputs.append({
                        "product": product,
                        "region": region,
                        "pass": False,
                        "check": "Metric Validity",
                        "explanation": f"LCOE misleading for declining system - {explanation_detail}",
                        "severity": "critical",
                        "recommendation": "DO NOT use LCOE for investment decisions on this technology (Adam's warning)"
                    })
                else:
                    outputs.append({
                        "product": product,
                        "region": region,
                        "pass": None,
                        "check": "Metric Validity",
                        "explanation": f"LCOE used for fossil fuel - interpret with caution",
                        "severity": "warning",
                        "recommendation": "LCOE may be misleading for fossil fuels in decline"
                    })

    return outputs

def regional_definition_validator(df: pd.DataFrame, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensures aggregated regions have clear definitions.
    Critical for understanding what's actually included in "Europe", "Asia", etc.
    """
    outputs = []

    # Aggregated regions that need definition
    ambiguous_regions = ['europe', 'asia', 'americas', 'africa', 'middle east', 'oecd', 'non-oecd']

    regions_found = set()
    for result in results:
        region = str(result.get('Region', '')).lower()
        if any(amb in region for amb in ambiguous_regions):
            regions_found.add(result.get('Region'))

    for region in regions_found:
        outputs.append({
            "product": "All",
            "region": region,
            "pass": None,
            "check": "Regional Definition",
            "explanation": f"'{region}' needs country list definition (e.g., does Europe include Russia?)",
            "severity": "info",
            "recommendation": f"Add metadata defining which countries are included in '{region}'"
        })

    return outputs

# ================================================================================
# OPTIONAL BUT USEFUL VALIDATORS
# ================================================================================

def unit_conversion_validator(df: pd.DataFrame,
                             conversions: Optional[Dict] = None,
                             tolerance: float = 0.05) -> List[Dict[str, Any]]:
    """
    Verify unit conversions are mathematically correct.
    Prevents basic calculation errors in multi-unit datasets.
    """
    outputs = []

    if conversions is None:
        conversions = {
            'boe_to_btu': 5.8e6,  # 1 BOE = 5.8 million BTU
            'kwh_to_btu': 3412,    # 1 kWh = 3412 BTU
            'liters_to_barrel': 159,  # 159 liters = 1 barrel
        }

    if df.empty:
        return outputs

    for (product, region, year), group in df.groupby(['product_name', 'region', 'year']):
        if len(group) > 1:
            units = group['unit'].tolist()
            values = group['value'].tolist()

            if any('btu' in str(u).lower() for u in units) and any('boe' in str(u).lower() for u in units):
                btu_val = next((v for v, u in zip(values, units) if 'btu' in str(u).lower()), None)
                boe_val = next((v for v, u in zip(values, units) if 'boe' in str(u).lower()), None)

                if btu_val and boe_val:
                    expected_boe = btu_val / conversions['boe_to_btu']
                    error = abs(expected_boe - boe_val) / boe_val if boe_val else 1.0

                    if error > tolerance:
                        outputs.append({
                            "product": product,
                            "region": region,
                            "pass": False,
                            "check": "Unit Conversion",
                            "explanation": f"BTU to BOE conversion off by {error:.1%} in year {year}"
                        })

    return outputs

def multi_source_consistency_validator(df_by_source: pd.DataFrame,
                                      threshold: float = 0.10) -> List[Dict[str, Any]]:
    """
    Compare data across IEA/EIA/BP/BNEF sources, flag >10% deviations.
    Useful for identifying measurement uncertainty.
    """
    outputs = []

    if df_by_source.empty or 'source' not in df_by_source.columns:
        return outputs

    key_cols = ['product_name', 'region', 'metric', 'year']
    if not all(col in df_by_source.columns for col in key_cols):
        return outputs

    for key, group in df_by_source.groupby(key_cols):
        if len(group) > 1:
            values = group['value'].dropna()
            sources = group['source'].tolist()

            if len(values) > 1:
                mean_val = values.mean()
                max_val = values.max()
                min_val = values.min()

                if mean_val > 0:
                    deviation = (max_val - min_val) / mean_val

                    if deviation > threshold:
                        outputs.append({
                            "product": key[0],
                            "region": key[1],
                            "pass": False,
                            "check": "Multi-Source Consistency",
                            "explanation": f"{deviation:.0%} deviation between sources: {', '.join(set(sources))}",
                            "evidence": {
                                "max": max_val,
                                "min": min_val,
                                "mean": mean_val,
                                "sources": sources
                            }
                        })

    return outputs


def regional_market_logic_validator(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Validates regional data makes logical sense based on known market characteristics.
    E.g., India should have significant 2-wheeler EV market, not just China.
    """
    outputs = []

    market_expectations = {
        'two wheeler (ev)': {
            'india': {'min_share': 0.15, 'reason': 'India has huge 2-wheeler market'},
            'china': {'max_share': 0.60, 'reason': 'China dominant but not exclusive'},
        },
        'passenger vehicle (ev)': {
            'china': {'min_share': 0.30, 'max_share': 0.60},
            'europe': {'min_share': 0.15, 'max_share': 0.35},
            'usa': {'min_share': 0.10, 'max_share': 0.25},
        }
    }

    for product_key, expectations in market_expectations.items():
        product_data = df[df['product_name'].str.lower().str.contains(product_key, na=False)]

        if not product_data.empty:
            latest_year = product_data['year'].max()
            latest_data = product_data[product_data['year'] == latest_year]

            total = latest_data['value'].sum()
            if total > 0:
                for region, limits in expectations.items():
                    region_value = latest_data[latest_data['region'].str.lower() == region]['value'].sum()
                    share = region_value / total

                    if 'min_share' in limits and share < limits['min_share']:
                        outputs.append({
                            "product": product_key,
                            "region": region,
                            "pass": False,
                            "check": "Regional Market Logic",
                            "explanation": f"{region} has only {share * 100:.1f}% share - seems too low",
                            "recommendation": limits.get('reason', 'Check regional market dynamics')
                        })

                    if 'max_share' in limits and share > limits['max_share']:
                        outputs.append({
                            "product": product_key,
                            "region": region,
                            "pass": False,
                            "check": "Regional Market Logic",
                            "explanation": f"{region} has {share * 100:.1f}% share - seems too high",
                            "recommendation": "Verify data source and regional definitions"
                        })

    return outputs


def definition_clarity_validator(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Flags metrics that need clear definitions based on expert feedback.
    """
    outputs = []

    ambiguous_terms = {
        'two wheeler': "Need to specify if includes motorcycles, e-bikes, or just bicycles",
        'recycling rate': "Specify if percentage of waste recycled or recovered copper",
        'annual deployment': "Clarify if new installations or cumulative capacity",
        'agentic ai': "Define what qualifies as 'agentic' vs traditional AI",
        'average lithium content': "Specify unit (kg per kWh or kg per vehicle)",
    }

    for term, definition_needed in ambiguous_terms.items():
        matching_data = df[df['metric'].str.lower().str.contains(term, na=False)]

        if not matching_data.empty:
            for product in matching_data['product_name'].unique():
                outputs.append({
                    "product": product,
                    "region": "All",
                    "pass": False,
                    "check": "Definition Clarity",
                    "explanation": definition_needed,
                    "severity": "medium"
                })

    return outputs


def cost_curve_anomaly_validator(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Catches illogical cost curves where global costs exceed regional costs.
    """
    outputs = []

    cost_data = df[df['unit'].str.contains('$|usd|cost', case=False, na=False)]

    for (product, metric, year), group in cost_data.groupby(['product_name', 'metric', 'year']):
        if 'global' in group['region'].str.lower().values:
            global_cost = group[group['region'].str.lower() == 'global']['value'].iloc[0]
            regional_costs = group[group['region'].str.lower() != 'global']['value'].values

            if len(regional_costs) > 0:
                min_regional = min(regional_costs)
                max_regional = max(regional_costs)

                if global_cost > max_regional * 1.1:
                    outputs.append({
                        "product": product,
                        "region": "Global",
                        "pass": False,
                        "check": "Cost Curve Anomaly",
                        "explanation": f"Global cost ({global_cost:.2f}) exceeds highest regional cost ({max_regional:.2f})",
                        "recommendation": "Global costs should reflect weighted average or economies of scale"
                    })

                if global_cost < min_regional * 0.5:
                    outputs.append({
                        "product": product,
                        "region": "Global",
                        "pass": False,
                        "check": "Cost Curve Anomaly",
                        "explanation": f"Global cost ({global_cost:.2f}) unrealistically below regional costs",
                        "recommendation": "Check if units are consistent across regions"
                    })

    return outputs


def ai_reality_check_validator(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Validates AI/GenAI deployment numbers against market reality.
    """
    outputs = []

    ai_data = df[df['product_name'].str.contains('agentic ai|generative ai|genai', case=False, na=False)]

    for (product, region), group in ai_data.groupby(['product_name', 'region']):
        if 'deployment' in group['metric'].iloc[0].lower():
            latest_year = group['year'].max()
            latest_value = group[group['year'] == latest_year]['value'].iloc[0]

            if latest_year <= 2025 and latest_value > 0.30:
                outputs.append({
                    "product": product,
                    "region": region,
                    "pass": False,
                    "check": "AI Reality Check",
                    "explanation": f"{latest_value * 100:.0f}% deployment in {latest_year} seems unrealistic",
                    "recommendation": "GenAI is nascent - 30%+ enterprise deployment unlikely before 2026"
                })

            if len(group) >= 2:
                group_sorted = group.sort_values('year')
                first_value = group_sorted.iloc[0]['value']
                years_diff = latest_year - group_sorted.iloc[0]['year']

                if years_diff > 0 and first_value > 0:
                    annual_growth = ((latest_value / first_value) ** (1 / years_diff)) - 1

                    if annual_growth > 1.0:
                        outputs.append({
                            "product": product,
                            "region": region,
                            "pass": False,
                            "check": "AI Reality Check",
                            "explanation": f"{annual_growth * 100:.0f}% annual growth rate unrealistic",
                            "recommendation": "Even breakthrough technologies rarely sustain >100% annual growth"
                        })

    return outputs


def adoption_curve_shape_validator(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Validates adoption curves follow expected S-curve pattern.
    Critical for catching unrealistic linear or exponential projections.
    """
    outputs = []

    adoption_metrics = ['adoption', 'deployment', 'penetration', 'fleet', 'sales']
    adoption_data = df[df['metric'].str.lower().str.contains('|'.join(adoption_metrics), na=False)]

    for (product, region), group in adoption_data.groupby(['product_name', 'region']):
        if len(group) >= 5:  # Need enough points to assess curve shape
            years = group['year'].values
            values = group['value'].values

            # Sort by year
            sorted_idx = np.argsort(years)
            years = years[sorted_idx]
            values = values[sorted_idx]

            # Calculate growth rates
            growth_rates = []
            for i in range(1, len(values)):
                if values[i - 1] > 0:
                    growth_rate = (values[i] - values[i - 1]) / values[i - 1]
                    growth_rates.append(growth_rate)

            if growth_rates:
                # Check for S-curve characteristics
                early_growth = np.mean(growth_rates[:len(growth_rates) // 3]) if len(growth_rates) >= 3 else 0
                late_growth = np.mean(growth_rates[-len(growth_rates) // 3:]) if len(growth_rates) >= 3 else 0

                # S-curves should show declining growth rates as they mature
                if late_growth > early_growth * 1.5 and values[-1] > 0.5:  # Still accelerating at high adoption
                    outputs.append({
                        "product": product,
                        "region": region,
                        "pass": False,
                        "check": "Adoption Curve Shape",
                        "explanation": "Adoption curve shows unrealistic acceleration at high penetration",
                        "recommendation": "S-curves should decelerate as they approach saturation"
                    })

    return outputs


def cost_curve_learning_rate_validator(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Validates cost curves follow realistic learning rates.
    Wright's Law: 20-30% cost reduction per doubling of cumulative production.
    """
    outputs = []

    cost_data = df[df['unit'].str.lower().str.contains('$|usd|cost|price', na=False)]

    for (product, region), group in cost_data.groupby(['product_name', 'region']):
        if len(group) >= 3:
            years = group['year'].values
            costs = group['value'].values

            # Calculate implied learning rate
            if costs[0] > 0 and costs[-1] > 0:
                years_elapsed = years[-1] - years[0]
                cost_reduction = (costs[0] - costs[-1]) / costs[0]
                annual_reduction = cost_reduction / years_elapsed if years_elapsed > 0 else 0

                # Check against reasonable bounds
                if 'solar' in product.lower() or 'battery' in product.lower():
                    if annual_reduction < 0.05:  # Less than 5% annual reduction
                        outputs.append({
                            "product": product,
                            "region": region,
                            "pass": False,
                            "check": "Cost Learning Rate",
                            "explanation": f"Only {annual_reduction * 100:.1f}% annual cost reduction - too slow for {product}",
                            "recommendation": "Technology costs should decline 10-20% annually in growth phase"
                        })
                    elif annual_reduction > 0.40:  # More than 40% annual reduction
                        outputs.append({
                            "product": product,
                            "region": region,
                            "pass": False,
                            "check": "Cost Learning Rate",
                            "explanation": f"{annual_reduction * 100:.1f}% annual reduction unrealistic",
                            "recommendation": "Even breakthrough technologies rarely exceed 30% annual cost reduction"
                        })

    return outputs


# ================================================================================
# ADDITIONAL VALIDATORS FROM BRADD'S SESSIONS (26-33)
# ================================================================================

def spelling_and_nomenclature_validator(results: List[Dict[str, Any]],
                                        df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Validator #26: Checks for common spelling errors.
    Based on actual issues Bradd identified.
    """
    outputs = []

    common_misspellings = {
        'artifical': 'artificial',
        'automous': 'autonomous',
        'lithiumion': 'lithium-ion',
        'passanger': 'passenger',
        'comercial': 'commercial',
        'telecome': 'telecom',
        'stationary storage': 'stationary storage',  # Often confused with stationery
    }

    checked = set()

    for result in results:
        entity = result.get('Entity_Name', '')
        if entity in checked:
            continue
        checked.add(entity)

        entity_lower = entity.lower()

        for wrong, correct in common_misspellings.items():
            if wrong in entity_lower:
                outputs.append({
                    'product': entity,
                    'region': result.get('Region', 'All'),
                    'pass': False,
                    'check': 'Spelling Check',
                    'explanation': f"Spelling: '{wrong}' should be '{correct}'",
                    'severity': 'low'
                })

    return outputs


def battery_size_evolution_validator(results: List[Dict[str, Any]],
                                     df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Validator #27: Validates battery sizes increasing as costs decrease.
    Bradd confirmed: lithium content per battery doubles over 10-15 years.
    """
    outputs = []

    for result in results:
        metric = result.get('Metric', '').lower()
        unit = result.get('Unit', '').lower()

        # Only check "lithium content per battery" metrics (not per kWh)
        if 'lithium content' in metric and 'per battery' in unit:
            years = result.get('X', [])
            values = result.get('Y', [])

            valid_data = [(y, v) for y, v in zip(years, values)
                          if v is not None and v > 0]

            if len(valid_data) >= 5:
                valid_data.sort(key=lambda x: x[0])

                # Compare early vs late periods
                early_avg = np.mean([v for _, v in valid_data[:2]])
                late_avg = np.mean([v for _, v in valid_data[-2:]])
                year_span = valid_data[-1][0] - valid_data[0][0]

                # Expect ~50-100% increase over 10+ years
                if year_span >= 10:
                    growth_ratio = late_avg / early_avg

                    if growth_ratio < 1.5:  # Less than 50% growth
                        outputs.append({
                            'product': result.get('Entity_Name'),
                            'region': result.get('Region'),
                            'pass': False,
                            'check': 'Battery Size Evolution',
                            'explanation': f"Battery size only grew {(growth_ratio - 1) * 100:.0f}% over {year_span} years - expected 50-100%"
                        })
                    else:
                        outputs.append({
                            'product': result.get('Entity_Name'),
                            'region': result.get('Region'),
                            'pass': True,
                            'check': 'Battery Size Evolution',
                            'explanation': f"Battery size appropriately increased {(growth_ratio - 1) * 100:.0f}%"
                        })

    return outputs


def regional_price_pattern_validator(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Validator #28: Validates developed vs developing market price patterns.
    Not exact ratios but general pattern: USA/Europe > China/India for vehicles.
    """
    outputs = []

    # Group regions by economic development
    developed_regions = ['usa', 'united states', 'europe', 'japan', 'canada']
    developing_regions = ['china', 'india', 'brazil', 'mexico']

    # Products where we expect price differences
    products_with_regional_pricing = ['vehicle', 'car', 'suv', 'sedan', 'hatchback']

    for product in products_with_regional_pricing:
        product_df = df[df['product_name'].str.lower().str.contains(product, na=False)]

        if product_df.empty:
            continue

        # Get latest year data
        latest_year = product_df['year'].max()
        latest_data = product_df[product_df['year'] == latest_year]

        # Calculate averages for each group
        developed_prices = []
        developing_prices = []

        for region in developed_regions:
            region_data = latest_data[latest_data['region'].str.lower().str.contains(region, na=False)]
            if not region_data.empty:
                developed_prices.extend(region_data['value'].tolist())

        for region in developing_regions:
            region_data = latest_data[latest_data['region'].str.lower().str.contains(region, na=False)]
            if not region_data.empty:
                developing_prices.extend(region_data['value'].tolist())

        if developed_prices and developing_prices:
            avg_developed = np.mean(developed_prices)
            avg_developing = np.mean(developing_prices)

            # Expect developed markets to be more expensive (but not exact ratio)
            if avg_developed < avg_developing:
                outputs.append({
                    'product': product,
                    'region': 'Cross-regional',
                    'pass': False,
                    'check': 'Regional Price Pattern',
                    'explanation': f"Developed markets (${avg_developed:.0f}) cheaper than developing (${avg_developing:.0f}) - unexpected",
                    'severity': 'medium'
                })
            elif avg_developed / avg_developing > 5:  # More than 5x seems excessive
                outputs.append({
                    'product': product,
                    'region': 'Cross-regional',
                    'pass': False,
                    'check': 'Regional Price Pattern',
                    'explanation': f"Price ratio {avg_developed / avg_developing:.1f}x seems excessive",
                    'severity': 'low'
                })

    return outputs


def historical_data_label_validator(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validator #29: Ensures historical data isn't labeled as projections.
    Bradd specifically noted this with data center/telecom growth.
    """
    outputs = []
    current_year = 2025

    projection_terms = ['projection', 'forecast', 'predicted', 'expected']

    for result in results:
        metric = result.get('Metric', '').lower()
        years = result.get('X', [])

        if any(term in metric for term in projection_terms):
            if years and max(years) <= current_year:
                outputs.append({
                    'product': result.get('Entity_Name'),
                    'region': result.get('Region'),
                    'pass': False,
                    'check': 'Historical Label',
                    'explanation': f"Data through {max(years)} labeled as 'projection' - should be 'historical'",
                    'severity': 'low'
                })

    return outputs


def growth_rate_reasonableness_validator(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Validator #30: Validates growth rates are reasonable for sector.
    Based on Bradd's feedback on data center (10-20%) and telecom (5-10%).
    """
    outputs = []

    sector_expectations = {
        'data center': {'typical_min': 10, 'typical_max': 20},
        'telecom': {'typical_min': 5, 'typical_max': 10},
        'ai': {'typical_min': 30, 'typical_max': 100},  # Higher for emerging tech
    }

    crisis_years = [2008, 2009, 2020, 2021]  # Allow exceptions

    for sector, bounds in sector_expectations.items():
        sector_df = df[df['product_name'].str.lower().str.contains(sector, na=False)]

        if not sector_df.empty and 'growth' in str(sector_df['metric'].iloc[0]).lower():
            for _, row in sector_df.iterrows():
                year = row['year']
                growth = row['value']

                # Skip crisis years
                if year in crisis_years:
                    continue

                if growth < bounds['typical_min'] or growth > bounds['typical_max']:
                    outputs.append({
                        'product': sector,
                        'region': row['region'],
                        'pass': False,
                        'check': 'Growth Rate Bounds',
                        'explanation': f"{growth:.1f}% growth in {year} outside typical {bounds['typical_min']}-{bounds['typical_max']}% range",
                        'severity': 'low'
                    })

    return outputs


def lithium_chemistry_validator(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validator #31: Validates lithium content follows chemistry constraints.
    Bradd: 100-150g lithium per kWh of battery capacity.
    """
    outputs = []

    # Group by entity to find related metrics
    entity_groups = {}
    for result in results:
        key = (result.get('Entity_Name'), result.get('Region'))
        if key not in entity_groups:
            entity_groups[key] = []
        entity_groups[key].append(result)

    for (entity, region), group in entity_groups.items():
        # Look for lithium and battery capacity metrics
        lithium_kg = None
        battery_kwh = None

        for result in group:
            metric_lower = result.get('Metric', '').lower()
            unit_lower = result.get('Unit', '').lower()

            if 'lithium content' in metric_lower and 'kg' in unit_lower:
                latest_values = result.get('Y', [])
                if latest_values:
                    lithium_kg = latest_values[-1]

            elif 'capacity' in metric_lower and 'kwh' in unit_lower:
                latest_values = result.get('Y', [])
                if latest_values:
                    battery_kwh = latest_values[-1]

        # Validate if we have both metrics
        if lithium_kg and battery_kwh and battery_kwh > 0:
            grams_per_kwh = (lithium_kg * 1000) / battery_kwh

            if grams_per_kwh < 100 or grams_per_kwh > 150:
                outputs.append({
                    'product': entity,
                    'region': region,
                    'pass': False,
                    'check': 'Lithium Chemistry',
                    'explanation': f"{grams_per_kwh:.0f}g/kWh outside chemical limits (100-150g/kWh)",
                    'severity': 'high'
                })

    return outputs


def commodity_price_behavior_validator(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validator #32: Ensures commodities show price volatility, not tech-like decline.
    Commodities don't follow Wright's Law - they follow supply/demand cycles.
    """
    outputs = []

    commodities = ['aluminum', 'copper', 'steel', 'iron', 'gold', 'silver', 'oil', 'gas']

    for result in results:
        entity_lower = result.get('Entity_Name', '').lower()
        unit_lower = result.get('Unit', '').lower()

        # Check if this is commodity price data
        is_commodity = any(comm in entity_lower for comm in commodities)
        is_price_data = any(ind in unit_lower for ind in ['$', 'usd', 'price', 'cost'])

        if is_commodity and is_price_data:
            years = result.get('X', [])
            values = result.get('Y', [])

            valid_data = [(y, v) for y, v in zip(years, values)
                          if v is not None and v > 0]

            if len(valid_data) >= 5:
                # Check for monotonic decline (tech-like behavior)
                declining_periods = sum(1 for i in range(len(valid_data) - 1)
                                        if valid_data[i][1] > valid_data[i + 1][1])

                decline_ratio = declining_periods / (len(valid_data) - 1)

                # If declining >80% of the time, it's too tech-like
                if decline_ratio > 0.8:
                    outputs.append({
                        'product': result.get('Entity_Name'),
                        'region': result.get('Region'),
                        'pass': False,
                        'check': 'Commodity Behavior',
                        'explanation': f"Commodity showing {decline_ratio * 100:.0f}% consistent decline - should show price cycles",
                        'severity': 'medium'
                    })

    return outputs


def trusted_source_validator(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validator #33: Checks for Bradd's explicitly trusted sources.
    """
    outputs = []

    # Sources Bradd specifically mentioned as reputable
    bradd_trusted = [
        'edmunds.com',  # Vehicle prices
        'kbb.com',  # Kelly's Blue Book
        'mckinsey.com',  # Analysis
        'usgs.gov',  # Materials
        'nvidia.com',  # Chips
    ]

    for result in results:
        entity = result.get('Entity_Name', '')
        urls = result.get('DataSource_URLs', [])

        if urls:
            trusted_found = []
            for url in urls:
                url_lower = str(url).lower()
                for trusted in bradd_trusted:
                    if trusted in url_lower:
                        trusted_found.append(trusted)

            # Special check for vehicle data
            if 'vehicle' in entity.lower() or any(v in entity.lower() for v in ['sedan', 'suv', 'hatchback']):
                if not any(s in str(urls).lower() for s in ['edmunds', 'kbb']):
                    outputs.append({
                        'product': entity,
                        'region': result.get('Region'),
                        'pass': False,
                        'check': 'Trusted Sources',
                        'explanation': "Vehicle data should use Edmunds or KBB per Bradd",
                        'severity': 'low'
                    })

    return outputs


def structural_break_detection_validator(results: List[Dict[str, Any]],
                                         df: pd.DataFrame,
                                         threshold_pct: float = 30.0) -> List[Dict[str, Any]]:
    """
    Validator #34: Detects structural breaks in time series (methodology changes, data source switches).
    Critical for identifying artificial jumps that aren't real market changes.
    """
    outputs = []

    for result in results:
        entity_name = result.get('Entity_Name', '')
        region = result.get('Region', '')
        years = result.get('X', [])
        values = result.get('Y', [])
        sources = result.get('DataSource_URLs', [])

        valid_data = [(y, v) for y, v in zip(years, values)
                      if v is not None and v > 0]

        if len(valid_data) < 5:
            continue

        valid_data.sort(key=lambda x: x[0])
        years_clean, values_clean = zip(*valid_data)

        source_change_years = []
        if len(set(sources)) > 1:
            source_change_years.append(years_clean[len(years_clean) // 2])

        breaks_detected = []
        window_size = min(3, len(values_clean) // 2)

        for i in range(window_size, len(values_clean) - window_size):
            before_mean = np.mean(values_clean[i - window_size:i])
            after_mean = np.mean(values_clean[i:i + window_size])

            if before_mean > 0:
                pct_change = abs((after_mean - before_mean) / before_mean) * 100

                if pct_change > threshold_pct:
                    if i > window_size and i < len(values_clean) - window_size - 1:
                        prev_change = abs(values_clean[i - 1] - values_clean[i - 2]) / values_clean[i - 2] if \
                        values_clean[i - 2] > 0 else 0
                        next_change = abs(values_clean[i + 1] - values_clean[i]) / values_clean[i] if values_clean[
                                                                                                          i] > 0 else 0

                        if pct_change > max(prev_change, next_change) * 3:
                            breaks_detected.append({
                                'year': years_clean[i],
                                'before_value': before_mean,
                                'after_value': after_mean,
                                'change_pct': pct_change
                            })

        if breaks_detected:
            for brk in breaks_detected:
                outputs.append({
                    'product': entity_name,
                    'region': region,
                    'pass': False,
                    'check': 'Structural Break Detection',
                    'explanation': f"Potential methodology change in {brk['year']}: {brk['change_pct']:.0f}% jump",
                    'severity': 'high',
                    'evidence': brk
                })
        elif len(set(sources)) > 1:
            outputs.append({
                'product': entity_name,
                'region': region,
                'pass': None,
                'check': 'Structural Break Detection',
                'explanation': f"Multiple data sources detected ({len(set(sources))}) - verify continuity",
                'severity': 'medium'
            })

    return outputs


def mass_balance_validator(df: pd.DataFrame,
                           tolerance: float = 0.05) -> List[Dict[str, Any]]:
    """
    Validator #35: Validates mass balance for commodities.
    """
    outputs = []

    if df.empty:
        return outputs

    commodities = ['aluminum', 'aluminium', 'copper', 'steel', 'iron', 'lithium']

    for commodity in commodities:
        commodity_df = df[df['product_name'].str.lower().str.contains(commodity, na=False)]

        if commodity_df.empty:
            continue

        for year in commodity_df['year'].unique():
            year_data = commodity_df[commodity_df['year'] == year]

            production = year_data[year_data['metric'].str.contains('production', case=False, na=False)]['value'].sum()
            consumption = year_data[year_data['metric'].str.contains('consumption', case=False, na=False)][
                'value'].sum()

            if production > 0 and consumption > 0:
                imbalance = abs(production - consumption) / production

                if imbalance > tolerance:
                    outputs.append({
                        'product': commodity,
                        'region': 'Global',
                        'pass': False,
                        'check': 'Mass Balance',
                        'explanation': f"Supply-demand imbalance of {imbalance * 100:.1f}%",
                        'severity': 'high' if imbalance > 0.1 else 'medium'
                    })

    return outputs
# ================================================================================
# HELPER FUNCTIONS
# ================================================================================
def run_all_domain_expert_validators(results: List[Dict[str, Any]],
                                     df: pd.DataFrame,
                                     config: Optional[Dict] = None) -> Dict[str, List[Dict]]:
    """
    Run ALL domain expert validators with complete coverage and return a dict:
        { <validator_key>: [ {product, region, pass, check, explanation, ...}, ... ], ... }

    Notes:
    - Uses your existing validator implementations in this module.
    - Handles mixed return types (DataFrame, list of strings, list of dicts, or None).
    - Leaves each validator's native 'check' label intact; the UI can group and display freely.
    """
    if config is None:
        config = {}

    all_validations: Dict[str, List[Dict]] = {}

    # ---- Full validator roster (35) ----
    validator_list: List[Tuple[str, Callable[[], Any]]] = [
        # Bradd’s higher-level curve checks
        ('bradd_cost_curve',       lambda: bradd_cost_curve_validator(results, df)),
        ('bradd_adoption_curve',   lambda: bradd_adoption_curve_validator(results, df)),

        # Data integrity (applies to cost & adoption)
        ('units_and_scale',        lambda: validate_units_and_scale(df)),
        ('year_anomalies',         lambda: check_year_anomalies(results, df)),
        ('definition_clarity',     lambda: definition_clarity_validator(df)),

        # Cost curve validators (Wright’s Law)
        ('cost_parity',            lambda: cost_parity_threshold(
                                        df,
                                        config.get('parity_level', 70),
                                        config.get('cutoff_year', 2022)
                                    )),
        ('cost_curve_anomaly',     lambda: cost_curve_anomaly_validator(df)),
        ('cost_curve_learning_rate', lambda: cost_curve_learning_rate_validator(df)),
        ('inflation_adjustment',   lambda: inflation_adjustment_validator(results, df)),

        # Adoption curve validators (S-curve)
        ('adoption_saturation',    lambda: adoption_saturation_feasibility(results)),
        ('adoption_curve_shape',   lambda: adoption_curve_shape_validator(df)),

        # Market context & anomalies (GFC 2008, COVID 2020/21, Ukraine 2022 explained)
        ('market_context',         lambda: market_context_validator(
                                        df,
                                        results,
                                        config.get('llm_client')
                                    )),

        # Energy transition (STDF core)
        ('oil_displacement',       lambda: oil_displacement_check(
                                        results,
                                        df,
                                        config.get('material_pct', 0.10),
                                        config.get('high_adoption', 0.70)
                                    )),
        ('derived_oil_displacement', lambda: derived_oil_displacement_validator(
                                        results,
                                        config.get('regional_params')
                                    )),
        ('global_oil_sanity',      lambda: global_oil_sanity_validator(
                                        df,
                                        config.get('transport_band', (55, 60)),
                                        config.get('total_band', (90, 110)),
                                        config.get('crisis_windows', [(2020, 2021)])
                                    )),

        # Capacity factor / stranded risk signal
        ('capacity_factor',        lambda: (
                                        capacity_factor_validator(
                                            df[df['metric'].str.contains('capacity',   case=False, na=False)],
                                            df[df['metric'].str.contains('generation', case=False, na=False)],
                                            config.get('decline_years', 3)
                                        )
                                    ) if (
                                        not df[df['metric'].str.contains('capacity',   case=False, na=False)].empty and
                                        not df[df['metric'].str.contains('generation', case=False, na=False)].empty
                                    ) else []),

        # Data-source and provenance
        ('data_source_quality',    lambda: data_source_quality_validator(results)),
        ('data_source_integrity',  lambda: data_source_integrity_validator(results)),
        ('multi_source_consistency', lambda: (
                                        multi_source_consistency_validator(df, config.get('consistency_threshold', 0.10))
                                    ) if 'source' in df.columns else []),

        # Regional/aggregation sanity checks
        ('regional_market_logic',  lambda: regional_market_logic_validator(df)),
        ('regional_definition',    lambda: regional_definition_validator(df, results)),
        ('global_sum',             lambda: global_sum_validator(df, config.get('sum_tolerance', 0.05))),

        # Other expert checks
        ('metric_validity',        lambda: metric_validity_validator(df)),
        ('unit_conversion',        lambda: unit_conversion_validator(
                                        df,
                                        config.get('conversions'),
                                        config.get('conversion_tolerance', 0.05)
                                    )),
        ('structural_break', lambda: structural_break_detection_validator(results, df, 30.0)),
        ('mass_balance', lambda: mass_balance_validator(df, 0.05)),
        # Add these 8 new validators to the list:
        ('spelling_nomenclature', lambda: spelling_and_nomenclature_validator(results, df)),
        ('battery_size_evolution', lambda: battery_size_evolution_validator(results, df)),
        ('regional_price_pattern', lambda: regional_price_pattern_validator(df)),
        ('historical_label', lambda: historical_data_label_validator(results)),
        ('growth_rate_bounds', lambda: growth_rate_reasonableness_validator(df)),
        ('lithium_chemistry', lambda: lithium_chemistry_validator(results)),
        ('commodity_behavior', lambda: commodity_price_behavior_validator(results)),
        ('trusted_sources', lambda: trusted_source_validator(results)),
        ('ai_reality',             lambda: ai_reality_check_validator(df)),
    ]

    # ---- Execute with robust handling ----
    for validator_name, validator_func in validator_list:
        try:
            logger.info(f"Running {validator_name} validator.")
            result = validator_func()

            # Normalize return shape → always a list[dict]
            if result is None:
                result = []
            elif isinstance(result, pd.DataFrame):
                # e.g., unit consistency may return a small DF of issues
                result = result.to_dict('records') if not result.empty else []
            elif isinstance(result, list) and result and isinstance(result[0], str):
                # e.g., year anomalies may return a list of strings
                result = [{
                    "product": "Multiple",
                    "region": "Multiple",
                    "pass": False,
                    "check": validator_name.replace('_', ' ').title(),
                    "explanation": msg
                } for msg in result]

            if result:
                all_validations[validator_name] = result
                logger.info(f"  - {validator_name}: {len(result)} issues/checks")
        except Exception as e:
            logger.error(f"{validator_name} validator failed: {e}")
            all_validations[validator_name] = [{
                "product": "Error",
                "region": "N/A",
                "pass": None,
                "check": validator_name.replace('_', ' ').title(),
                "explanation": f"Validator error: {str(e)[:100]}"
            }]

    return all_validations


def format_expert_validation_summary(all_validations: Dict[str, List[Dict]]) -> str:
    """
    Format validation results into readable summary
    """
    lines = []
    lines.append("=" * 80)
    lines.append("DOMAIN EXPERT VALIDATION SUMMARY")
    lines.append("=" * 80)

    total_checks = 0
    total_passed = 0
    total_failed = 0
    total_info = 0

    for validator_name, results in all_validations.items():
        if not results:
            continue

        passed = sum(1 for r in results if r.get('pass') is True)
        failed = sum(1 for r in results if r.get('pass') is False)
        info_only = sum(1 for r in results if r.get('pass') is None)

        total_checks += len(results)
        total_passed += passed
        total_failed += failed
        total_info += info_only

        lines.append(f"\n{validator_name.replace('_', ' ').title()}:")
        lines.append(f"  ✅ Passed: {passed}")
        lines.append(f"  ❌ Failed: {failed}")
        if info_only > 0:
            lines.append(f"  ℹ️ Info: {info_only}")

        # Show critical failures
        critical = [r for r in results if r.get('severity') == 'critical']
        for crit in critical[:2]:
            lines.append(f"    🚨 CRITICAL: {crit['product']} - {crit['explanation']}")

        # Show other failures
        failures = [r for r in results if r.get('pass') is False and r.get('severity') != 'critical']
        for fail in failures[:2]:
            lines.append(f"    - {fail['product']} ({fail['region']}): {fail['explanation']}")

        if len(failures) > 2:
            lines.append(f"    ... and {len(failures) - 2} more issues")

    lines.append("\n" + "=" * 80)
    lines.append("OVERALL SUMMARY")
    lines.append(f"Total Checks: {total_checks}")
    lines.append(f"Passed: {total_passed} ({total_passed/total_checks*100:.1f}%)" if total_checks > 0 else "Passed: 0")
    lines.append(f"Failed: {total_failed} ({total_failed/total_checks*100:.1f}%)" if total_checks > 0 else "Failed: 0")
    lines.append(f"Info/Warnings: {total_info}")
    lines.append("=" * 80)

    return "\n".join(lines)