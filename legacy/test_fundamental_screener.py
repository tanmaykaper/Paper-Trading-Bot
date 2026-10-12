# test_fundamental_screener.py
# Test Module 3: Fundamental Screener

from fundamental_screener import FundamentalScreener

print("\n" + "="*70)
print("TESTING MODULE 3: FUNDAMENTAL SCREENER")
print("="*70)

screener = FundamentalScreener()

# Test 1: Stock that PASSES (good fundamentals)
print("\n✓ Test 1: Stock with GOOD Fundamentals (Should PASS)")
good_fund = {
    'pe_ratio': 22.0,
    'sector_avg_pe': 25.0,
    'debt_to_equity': 0.65,
    'roe_5yr': 0.18,
    'revenue_cagr': 0.12,
    'current_ratio': 1.4,
}

passed, checks = screener.check_fundamental_gate(good_fund)
print(f"  Result: {'✓ PASSED' if passed else '✗ FAILED'}")
print(f"  Checks: {checks}")
assert passed == True, "Good stock should pass"
print(screener.get_check_summary('RELIANCE', good_fund, checks))

# Test 2: Stock with HIGH P/E (Should FAIL)
print("\n✓ Test 2: Stock with HIGH P/E (Should FAIL)")
bad_pe_fund = {
    'pe_ratio': 40.0,  # > 30 (sector avg × 1.2)
    'sector_avg_pe': 25.0,
    'debt_to_equity': 0.65,
    'roe_5yr': 0.18,
    'revenue_cagr': 0.12,
    'current_ratio': 1.4,
}

passed, checks = screener.check_fundamental_gate(bad_pe_fund)
print(f"  Result: {'✓ PASSED' if passed else '✗ FAILED'}")
assert passed == False, "High P/E stock should fail"
assert checks['pe_check'] == False, "P/E check should fail"
print("  ✓ Correctly rejected high P/E stock")
print(screener.get_check_summary('OVERVALUED', bad_pe_fund, checks))

# Test 3: Stock with HIGH DEBT (Should FAIL)
print("\n✓ Test 3: Stock with HIGH Debt (Should FAIL)")
bad_de_fund = {
    'pe_ratio': 22.0,
    'sector_avg_pe': 25.0,
    'debt_to_equity': 0.90,  # > 0.75
    'roe_5yr': 0.18,
    'revenue_cagr': 0.12,
    'current_ratio': 1.4,
}

passed, checks = screener.check_fundamental_gate(bad_de_fund)
print(f"  Result: {'✓ PASSED' if passed else '✗ FAILED'}")
assert passed == False, "High debt stock should fail"
assert checks['de_check'] == False, "D/E check should fail"
print("  ✓ Correctly rejected high debt stock")

# Test 4: Stock with LOW ROE (Should FAIL)
print("\n✓ Test 4: Stock with LOW ROE (Should FAIL)")
bad_roe_fund = {
    'pe_ratio': 22.0,
    'sector_avg_pe': 25.0,
    'debt_to_equity': 0.65,
    'roe_5yr': 0.10,  # < 0.15
    'revenue_cagr': 0.12,
    'current_ratio': 1.4,
}

passed, checks = screener.check_fundamental_gate(bad_roe_fund)
print(f"  Result: {'✓ PASSED' if passed else '✗ FAILED'}")
assert passed == False, "Low ROE stock should fail"
assert checks['roe_check'] == False, "ROE check should fail"
print("  ✓ Correctly rejected low ROE stock")

# Test 5: Borderline stock (at all thresholds)
print("\n✓ Test 5: Stock at ALL Thresholds (Should PASS - inclusive)")
borderline_fund = {
    'pe_ratio': 30.0,  # Exactly at sector avg × 1.2
    'sector_avg_pe': 25.0,
    'debt_to_equity': 0.75,  # Exactly at threshold
    'roe_5yr': 0.15,  # Exactly at 15%
    'revenue_cagr': 0.10,  # Exactly at 10%
    'current_ratio': 1.2,  # Exactly at threshold
}

passed, checks = screener.check_fundamental_gate(borderline_fund)
print(f"  Result: {'✓ PASSED' if passed else '✗ FAILED'}")
assert passed == True, "Borderline stock should pass (thresholds inclusive)"
print("  ✓ Correctly accepted borderline stock")

print("\n" + "="*70)
print("✅ MODULE 3 TESTS COMPLETE")
print("="*70 + "\n")
