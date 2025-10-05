#!/usr/bin/env python3
"""
Universal Calibration Module
Calibration engine that adapts to any data type using multiple validation sources
"""

import statistics
import random
import re
from math import log10, floor
from typing import List, Dict, Tuple, Optional, Any
from models import ValidationPoint, CalibrationResult, ScaleInfo
from config import (
    ECONOMIC_INDICATORS, UNIT_MAPPINGS, LLM_TEMPERATURE, 
    LLM_MAX_TOKENS_VALIDATION
)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class UniversalCalibrator:
    """Universal calibration engine that adapts to any data type"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.validation_cache: Dict[str, List[ValidationPoint]] = {}
        self.regional_economic_data = ECONOMIC_INDICATORS
    
    async def calibrate_values(
        self, 
        values: List[float], 
        query_description: str,
        unit: str, 
        region: str = 'global'
    ) -> CalibrationResult:
        """
        Universal calibration method for any data type
        
        Functional pipeline: values -> analysis -> adjustment -> result
        """
        if not values or len(values) < 2:
            return self._create_empty_result(values)
        
        # Analysis pipeline
        scale_info = self._get_unit_scale_intelligence(unit, values)
        validation_points = await self._get_llm_validation_ensemble(
            query_description, unit, region
        )
        reference_range = self._create_reference_range(
            values, scale_info, validation_points, region
        )
        
        # Determine adjustment
        adjustment_result = self._calculate_adjustment(
            values, reference_range, scale_info
        )
        
        # Apply calibration if needed
        calibrated_values = (
            self._apply_intelligent_adjustment(
                values, 
                adjustment_result['factor'], 
                scale_info
            ) if adjustment_result['needs_adjustment']
            else values.copy()
        )
        
        return CalibrationResult(
            original_values=values,
            calibrated_values=calibrated_values,
            adjustment_factor=adjustment_result['factor'],
            confidence_score=adjustment_result['confidence'],
            validation_method=adjustment_result['method'],
            reference_range=reference_range,
            adjustment_reasoning=adjustment_result['reasoning']
        )
    
    def _create_empty_result(self, values: List[float]) -> CalibrationResult:
        """Create result for insufficient data"""
        return CalibrationResult(
            original_values=values,
            calibrated_values=values,
            adjustment_factor=1.0,
            confidence_score=0.0,
            validation_method='insufficient_data',
            reference_range=(0, 0)
        )
    
    def _get_unit_scale_intelligence(
        self, 
        unit: str, 
        values: List[float]
    ) -> ScaleInfo:
        """
        Intelligent unit scale analysis for any unit type
        
        Pure function that analyzes unit characteristics
        """
        unit_lower = unit.lower()
        avg_value = statistics.mean(values) if values else 100
        
        magnitude_order = floor(log10(abs(avg_value))) if avg_value > 0 else 2
        
        # Detect unit type from mappings
        unit_type, expected_range, volatility = self._detect_unit_type(unit_lower)
        
        growth_expectation = 0.15 if unit_type == 'power' else 0.05
        
        return ScaleInfo(
            magnitude_order=magnitude_order,
            unit_type=unit_type,
            expected_range=expected_range,
            volatility_factor=volatility,
            growth_expectation=growth_expectation
        )
    
    def _detect_unit_type(
        self, 
        unit_lower: str
    ) -> Tuple[str, Tuple[float, float], float]:
        """Detect unit type from mappings"""
        for unit_type, (keywords, expected_range, volatility) in UNIT_MAPPINGS.items():
            if any(keyword in unit_lower for keyword in keywords):
                return unit_type, expected_range, volatility
        
        return 'unknown', (1, 1000), 0.1
    
    async def _get_llm_validation_ensemble(
        self, 
        query_description: str,
        unit: str, 
        region: str
    ) -> List[ValidationPoint]:
        """
        Get validation from multiple LLM models for ensemble averaging
        
        Returns list of validation points from different models
        """
        if not self.api_key or not GROQ_AVAILABLE:
            return []
        
        validation_points = []
        
        try:
            client = Groq(api_key=self.api_key)
            prompt = self._create_validation_prompt(query_description, unit, region)
            
            models = ["llama-3.1-8b-instant", "llama3-groq-70b-8192-tool-use-preview"]
            
            for model in models:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=LLM_TEMPERATURE,
                        max_tokens=LLM_MAX_TOKENS_VALIDATION
                    )
                    
                    validation_point = self._parse_llm_validation(
                        response.choices[0].message.content, 
                        model
                    )
                    
                    if validation_point:
                        validation_points.append(validation_point)
                        
                except Exception:
                    continue
                    
        except Exception:
            pass
        
        return validation_points
    
    def _create_validation_prompt(
        self, 
        query_description: str,
        unit: str, 
        region: str
    ) -> str:
        """Create adaptive validation prompt for any data type"""
        return f"""You are a data analyst. Provide a realistic range for: {query_description}

Context:
- Unit: {unit}
- Region: {region}
- Current year: 2024

Consider:
- Market conditions and economic factors
- Regional differences (cost of living, development level, market maturity)
- Industry standards and typical scales
- Recent trends and realistic growth patterns

Respond ONLY with: MIN_VALUE-MAX_VALUE
Example: 15000-25000

Range: """
    
    def _parse_llm_validation(
        self, 
        llm_response: str, 
        model: str
    ) -> Optional[ValidationPoint]:
        """Parse LLM response to extract validation range"""
        try:
            range_match = re.search(
                r'(\d+(?:\.\d+)?(?:,\d+)*)\s*[-–]\s*(\d+(?:\.\d+)?(?:,\d+)*)', 
                llm_response
            )
            
            if not range_match:
                return None
            
            min_val = float(range_match.group(1).replace(',', ''))
            max_val = float(range_match.group(2).replace(',', ''))
            
            if min_val < max_val and min_val > 0:
                mid_val = (min_val + max_val) / 2
                confidence = 0.8 if 'groq-70b' in model else 0.7
                
                return ValidationPoint(
                    value=mid_val,
                    year=None,
                    source_type='llm',
                    confidence=confidence,
                    context=f"Range: {min_val}-{max_val} from {model}"
                )
        except Exception:
            pass
        
        return None
    
    def _create_reference_range(
        self,
        values: List[float],
        scale_info: ScaleInfo,
        validation_points: List[ValidationPoint],
        region: str
    ) -> Tuple[float, float]:
        """Create intelligent reference range from multiple sources"""
        # Apply regional intelligence to base range
        regional_min = self._apply_regional_intelligence(
            scale_info.expected_range[0], 
            region, 
            scale_info.unit_type
        )
        regional_max = self._apply_regional_intelligence(
            scale_info.expected_range[1], 
            region, 
            scale_info.unit_type
        )
        
        reference_ranges = [(regional_min, regional_max)]
        
        # Incorporate LLM validations
        for vp in validation_points:
            range_width = vp.value * 0.4
            llm_min = (vp.value - range_width) * vp.confidence
            llm_max = (vp.value + range_width) * vp.confidence
            reference_ranges.append((llm_min, llm_max))
        
        # Average multiple ranges
        if len(reference_ranges) > 1:
            avg_min = statistics.mean([r[0] for r in reference_ranges])
            avg_max = statistics.mean([r[1] for r in reference_ranges])
        else:
            avg_min, avg_max = reference_ranges[0]
        
        return (max(0, avg_min), max(avg_min * 2, avg_max))
    
    def _apply_regional_intelligence(
        self, 
        base_value: float,
        region: str, 
        unit_type: str
    ) -> float:
        """Apply intelligent regional adjustments based on economic factors"""
        if not region:
            return base_value
        
        region_key = region.lower().replace(' ', '_')
        economic_factor = self.regional_economic_data.get(region_key, 0.5)
        
        # PPP adjustment for cost-related metrics
        if unit_type in ['monetary', 'energy']:
            ppp_key = f'ppp_{region_key}'
            if ppp_key in self.regional_economic_data:
                economic_factor = self.regional_economic_data[ppp_key]
        
        # Technology adoption adjustments
        if unit_type in ['power', 'energy', 'quantity']:
            tech_factors = {
                'china': 1.5, 'usa': 1.0, 'germany': 0.8, 
                'japan': 0.7, 'india': 1.2
            }
            tech_factor = tech_factors.get(region_key, 1.0)
            return base_value * economic_factor * tech_factor
        
        return base_value * economic_factor
    
    def _calculate_adjustment(
        self,
        values: List[float],
        reference_range: Tuple[float, float],
        scale_info: ScaleInfo
    ) -> Dict[str, Any]:
        """Calculate adjustment needs and parameters"""
        current_avg = statistics.mean(values)
        reference_avg = (reference_range[0] + reference_range[1]) / 2
        
        lower_bound = reference_range[0] * 0.3
        upper_bound = reference_range[1] * 2.0
        
        reasoning = []
        
        if current_avg < lower_bound:
            return {
                'needs_adjustment': True,
                'factor': reference_avg / current_avg,
                'method': 'scale_up_significant',
                'confidence': 0.85,
                'reasoning': ["Values significantly below expected range"]
            }
        
        if current_avg > upper_bound:
            return {
                'needs_adjustment': True,
                'factor': reference_avg / current_avg,
                'method': 'scale_down_significant',
                'confidence': 0.85,
                'reasoning': ["Values significantly above expected range"]
            }
        
        if min(values) < reference_range[0] or max(values) > reference_range[1]:
            return {
                'needs_adjustment': True,
                'factor': self._calculate_bounded_adjustment(values, reference_range),
                'method': 'bounded_correction',
                'confidence': 0.75,
                'reasoning': ["Some values outside realistic bounds"]
            }
        
        return {
            'needs_adjustment': False,
            'factor': 1.0,
            'method': 'validated_realistic',
            'confidence': 0.9,
            'reasoning': ["Values within expected realistic range"]
        }
    
    def _calculate_bounded_adjustment(
        self, 
        values: List[float],
        reference_range: Tuple[float, float]
    ) -> float:
        """Calculate adjustment factor for bounded correction"""
        adjusted_values = [
            self._bound_value(val, reference_range)
            for val in values
        ]
        
        original_sum = sum(values)
        adjusted_sum = sum(adjusted_values)
        
        return adjusted_sum / original_sum if original_sum != 0 else 1.0
    
    def _bound_value(
        self, 
        value: float,
        reference_range: Tuple[float, float]
    ) -> float:
        """Bound single value to reference range with randomness"""
        if value < reference_range[0]:
            return reference_range[0] * random.uniform(1.0, 1.1)
        elif value > reference_range[1]:
            return reference_range[1] * random.uniform(0.9, 1.0)
        else:
            return value
    
    def _apply_intelligent_adjustment(
        self,
        values: List[float],
        adjustment_factor: float,
        scale_info: ScaleInfo
    ) -> List[float]:
        """Apply adjustment while preserving realistic patterns"""
        volatility = scale_info.volatility_factor
        
        adjusted_values = []
        for i, val in enumerate(values):
            adjusted_val = val * adjustment_factor
            
            # Add realistic variation for significant adjustments
            if i > 0 and abs(adjustment_factor - 1.0) > 0.2:
                variation = random.uniform(-volatility, volatility)
                adjusted_val *= (1 + variation)
            
            adjusted_values.append(max(0.001, adjusted_val))
        
        return adjusted_values


# Standalone API function
async def calibrate_any_data(
    values: List[float],
    query_description: str,
    unit: str,
    region: str = 'global',
    api_key: Optional[str] = None
) -> CalibrationResult:
    """
    Universal function to calibrate any data type - standalone API
    
    Pure functional interface to calibration system
    """
    calibrator = UniversalCalibrator(api_key)
    return await calibrator.calibrate_values(
        values, 
        query_description, 
        unit, 
        region
    )