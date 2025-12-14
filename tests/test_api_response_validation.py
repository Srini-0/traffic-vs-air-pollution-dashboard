"""
Property-based tests for API response validation.

**Feature: traffic-pollution-dashboard, Property 10: API response validation**
"""

import pytest
from hypothesis import given, strategies as st, settings as hypothesis_settings, HealthCheck
from unittest.mock import Mock, patch
import json

from traffic_pollution_dashboard.data.traffic_client import TrafficDataClient
from traffic_pollution_dashboard.data.pollution_client import PollutionDataClient
from traffic_pollution_dashboard.data.exceptions import DataValidationError


class TestAPIResponseValidation:
    """Property-based tests for API response validation across all clients."""
    
    def create_traffic_client(self):
        """Create a TrafficDataClient instance for testing."""
        return TrafficDataClient(api_key="test_key", base_url="https://test-api.com")
    
    def create_pollution_client(self):
        """Create a PollutionDataClient instance for testing."""
        return PollutionDataClient(api_key="test_key", base_url="https://test-api.com")
    
    @given(
        response_data=st.one_of(
            # Valid response structures
            st.fixed_dictionaries({
                'data': st.lists(st.dictionaries(
                    keys=st.text(min_size=1, max_size=20),
                    values=st.one_of(st.integers(), st.floats(), st.text())
                )),
                'status': st.sampled_from(['success', 'ok', 'completed'])
            }),
            # Invalid response structures
            st.one_of(
                st.none(),
                st.text(),
                st.integers(),
                st.lists(st.integers()),
                st.dictionaries(
                    keys=st.text(min_size=1, max_size=10),
                    values=st.one_of(st.integers(), st.text())
                )
            )
        )
    )
    @hypothesis_settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_traffic_api_response_validation_robustness(self, response_data):
        """
        Property test: For any API response received, the traffic client should validate 
        the data schema and handle malformed responses without system failure.
        
        **Feature: traffic-pollution-dashboard, Property 10: API response validation**
        **Validates: Requirements 6.2**
        """
        client = self.create_traffic_client()
        
        # Act - validate response (should never crash)
        try:
            result = client.validate_response(response_data)
            
            # Assert - result should always be a boolean
            assert isinstance(result, bool), "validate_response should always return a boolean"
            
            # If response_data is a dict with required fields, should return True
            if (isinstance(response_data, dict) and 
                'data' in response_data and 
                'status' in response_data):
                assert result == True, "Valid response structure should return True"
            else:
                # Invalid structures should return False
                assert result == False, "Invalid response structure should return False"
                
        except Exception as e:
            # Validation should never raise exceptions - it should return False instead
            pytest.fail(f"validate_response should not raise exceptions, got: {type(e).__name__}: {e}")
    
    @given(
        response_data=st.one_of(
            # Valid pollution API response structures
            st.fixed_dictionaries({
                'list': st.lists(st.dictionaries(
                    keys=st.sampled_from(['dt', 'main', 'components']),
                    values=st.one_of(st.integers(), st.dictionaries(
                        keys=st.text(min_size=1, max_size=10),
                        values=st.floats()
                    ))
                ))
            }),
            # Invalid response structures
            st.one_of(
                st.none(),
                st.text(),
                st.integers(),
                st.lists(st.integers()),
                st.dictionaries(
                    keys=st.text(min_size=1, max_size=10),
                    values=st.one_of(st.integers(), st.text())
                )
            )
        )
    )
    @hypothesis_settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_pollution_api_response_validation_robustness(self, response_data):
        """
        Property test: For any API response received, the pollution client should validate 
        the data schema and handle malformed responses without system failure.
        
        **Feature: traffic-pollution-dashboard, Property 10: API response validation**
        **Validates: Requirements 6.2**
        """
        client = self.create_pollution_client()
        
        # Act - validate response (should never crash)
        try:
            result = client.validate_response(response_data)
            
            # Assert - result should always be a boolean
            assert isinstance(result, bool), "validate_response should always return a boolean"
            
            # If response_data is a dict with required fields, should return True
            if isinstance(response_data, dict) and 'list' in response_data:
                assert result == True, "Valid response structure should return True"
            else:
                # Invalid structures should return False
                assert result == False, "Invalid response structure should return False"
                
        except Exception as e:
            # Validation should never raise exceptions - it should return False instead
            pytest.fail(f"validate_response should not raise exceptions, got: {type(e).__name__}: {e}")
    
    @given(
        malformed_json=st.one_of(
            st.text(min_size=1, max_size=100),  # Random text
            st.text().filter(lambda x: not x.strip()),  # Whitespace only
            st.just('{"incomplete": json'),  # Incomplete JSON
            st.just('[1, 2, 3,]'),  # Trailing comma
            st.just('{"key": }'),  # Missing value
            st.just('null'),  # Valid JSON but not expected structure
            st.just('[]'),  # Empty array
            st.just('{}'),  # Empty object
        )
    )
    @hypothesis_settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_malformed_json_handling(self, malformed_json):
        """
        Property test: For any malformed JSON string, the system should handle 
        parsing errors gracefully without crashing.
        
        **Feature: traffic-pollution-dashboard, Property 10: API response validation**
        **Validates: Requirements 6.2**
        """
        traffic_client = self.create_traffic_client()
        pollution_client = self.create_pollution_client()
        
        # Try to parse the malformed JSON
        try:
            parsed_data = json.loads(malformed_json)
            
            # If parsing succeeds, validation should still work
            traffic_result = traffic_client.validate_response(parsed_data)
            pollution_result = pollution_client.validate_response(parsed_data)
            
            assert isinstance(traffic_result, bool), "Traffic validation should return boolean"
            assert isinstance(pollution_result, bool), "Pollution validation should return boolean"
            
        except json.JSONDecodeError:
            # This is expected for malformed JSON - the system should handle this gracefully
            # In a real scenario, this would be caught at the HTTP response parsing level
            pass
        except Exception as e:
            # Any other exception is unexpected
            pytest.fail(f"Unexpected exception handling malformed JSON: {type(e).__name__}: {e}")
    
    def test_response_validation_with_missing_required_fields(self):
        """
        Test response validation with systematically missing required fields.
        
        **Feature: traffic-pollution-dashboard, Property 10: API response validation**
        **Validates: Requirements 6.2**
        """
        traffic_client = self.create_traffic_client()
        pollution_client = self.create_pollution_client()
        
        # Test traffic API validation
        traffic_test_cases = [
            {},  # Empty dict
            {'data': []},  # Missing status
            {'status': 'success'},  # Missing data
            {'data': [], 'status': 'success'},  # Valid
            {'data': None, 'status': 'success'},  # Null data
            {'data': [], 'status': None},  # Null status
        ]
        
        expected_traffic_results = [False, False, False, True, True, True]
        
        for i, test_case in enumerate(traffic_test_cases):
            result = traffic_client.validate_response(test_case)
            assert result == expected_traffic_results[i], \
                f"Traffic validation failed for case {i}: {test_case}"
        
        # Test pollution API validation
        pollution_test_cases = [
            {},  # Empty dict
            {'list': []},  # Valid
            {'data': []},  # Wrong field name
            {'list': None},  # Null list
            {'list': [{'dt': 123, 'components': {}}]},  # Valid structure
        ]
        
        expected_pollution_results = [False, True, False, True, True]
        
        for i, test_case in enumerate(pollution_test_cases):
            result = pollution_client.validate_response(test_case)
            assert result == expected_pollution_results[i], \
                f"Pollution validation failed for case {i}: {test_case}"
    
    @given(
        response_size=st.integers(min_value=0, max_value=10000),
        field_count=st.integers(min_value=0, max_value=100)
    )
    @hypothesis_settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_large_response_validation_performance(self, response_size, field_count):
        """
        Property test: Response validation should handle large responses efficiently 
        without performance degradation or memory issues.
        
        **Feature: traffic-pollution-dashboard, Property 10: API response validation**
        **Validates: Requirements 6.2**
        """
        import time
        
        traffic_client = self.create_traffic_client()
        pollution_client = self.create_pollution_client()
        
        # Create large response data
        large_data = []
        for i in range(response_size):
            item = {}
            for j in range(min(field_count, 20)):  # Limit field count to avoid excessive memory
                item[f'field_{j}'] = f'value_{i}_{j}'
            large_data.append(item)
        
        traffic_response = {'data': large_data, 'status': 'success'}
        pollution_response = {'list': large_data}
        
        # Test validation performance
        start_time = time.time()
        traffic_result = traffic_client.validate_response(traffic_response)
        traffic_time = time.time() - start_time
        
        start_time = time.time()
        pollution_result = pollution_client.validate_response(pollution_response)
        pollution_time = time.time() - start_time
        
        # Assert validation completes and returns correct results
        assert traffic_result == True, "Large traffic response should be valid"
        assert pollution_result == True, "Large pollution response should be valid"
        
        # Assert reasonable performance (should complete within 1 second for any reasonable size)
        assert traffic_time < 1.0, f"Traffic validation took too long: {traffic_time:.3f}s"
        assert pollution_time < 1.0, f"Pollution validation took too long: {pollution_time:.3f}s"
    
    def test_response_validation_with_unicode_and_special_characters(self):
        """
        Test response validation with unicode and special characters.
        
        **Feature: traffic-pollution-dashboard, Property 10: API response validation**
        **Validates: Requirements 6.2**
        """
        traffic_client = self.create_traffic_client()
        pollution_client = self.create_pollution_client()
        
        # Test with unicode and special characters
        unicode_test_cases = [
            {'data': [{'city': 'दिल्ली', 'value': 123}], 'status': 'success'},  # Hindi
            {'data': [{'city': 'বেঙ্গালুরু', 'value': 456}], 'status': 'success'},  # Bengali
            {'data': [{'special': '!@#$%^&*()', 'value': 789}], 'status': 'success'},  # Special chars
            {'list': [{'location': 'चेन्नई', 'aqi': 100}]},  # Hindi for pollution
        ]
        
        # All should be valid regardless of character encoding
        assert traffic_client.validate_response(unicode_test_cases[0]) == True
        assert traffic_client.validate_response(unicode_test_cases[1]) == True
        assert traffic_client.validate_response(unicode_test_cases[2]) == True
        assert pollution_client.validate_response(unicode_test_cases[3]) == True
    
    @given(
        nested_depth=st.integers(min_value=1, max_value=10)
    )
    @hypothesis_settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_deeply_nested_response_validation(self, nested_depth):
        """
        Property test: Response validation should handle deeply nested structures 
        without stack overflow or excessive recursion.
        
        **Feature: traffic-pollution-dashboard, Property 10: API response validation**
        **Validates: Requirements 6.2**
        """
        traffic_client = self.create_traffic_client()
        
        # Create deeply nested structure
        nested_data = {'value': 'leaf'}
        for i in range(nested_depth):
            nested_data = {'level': i, 'nested': nested_data}
        
        response = {'data': [nested_data], 'status': 'success'}
        
        # Should handle nested structures without issues
        try:
            result = traffic_client.validate_response(response)
            assert isinstance(result, bool), "Should return boolean even for deeply nested data"
            assert result == True, "Valid nested structure should return True"
        except RecursionError:
            pytest.fail("Response validation should not cause stack overflow on nested data")
        except Exception as e:
            pytest.fail(f"Unexpected error with nested data: {type(e).__name__}: {e}")