# 그룹 내부 신호 전수조사

## SEARCH (n=28782, 클래스 ['glob_pattern', 'grep_search', 'list_directory', 'read_file'])
최빈 찍기 정확도: 0.344
| 피처 | 정규화 MI |
|---|---|
| last2_path | 0.2078 |
| last_action | 0.0491 |
| turn | 0.0459 |
| step | 0.0459 |
| hist_len | 0.0458 |
| open_n | 0.0403 |
| last_group | 0.0368 |
| elapsed_b | 0.0156 |
| au | 0.0141 |
| last_result | 0.0093 |
| dirty | 0.0038 |
| budget_b | 0.0013 |
| mix | 0.0009 |
| ci | 0.0008 |
| loc_b | 0.0003 |

## MODIFY (n=17475, 클래스 ['apply_patch', 'edit_file', 'write_file'])
최빈 찍기 정확도: 0.639
| 피처 | 정규화 MI |
|---|---|
| last2_path | 0.3343 |
| last_action | 0.2782 |
| open_n | 0.2184 |
| turn | 0.2063 |
| step | 0.2063 |
| hist_len | 0.2060 |
| last_group | 0.1407 |
| last_result | 0.0661 |
| elapsed_b | 0.0600 |
| dirty | 0.0512 |
| ci | 0.0067 |
| budget_b | 0.0036 |
| au | 0.0032 |
| prompt_len | 0.0025 |
| mix | 0.0011 |

## EXEC (n=11912, 클래스 ['lint_or_typecheck', 'run_bash', 'run_tests'])
최빈 찍기 정확도: 0.425
| 피처 | 정규화 MI |
|---|---|
| last2_path | 0.2444 |
| last_action | 0.1835 |
| turn | 0.0990 |
| step | 0.0990 |
| hist_len | 0.0980 |
| last_group | 0.0948 |
| open_n | 0.0852 |
| mix | 0.0776 |
| last_result | 0.0690 |
| dirty | 0.0529 |
| elapsed_b | 0.0334 |
| ci | 0.0080 |
| loc_b | 0.0039 |
| budget_b | 0.0030 |
| au | 0.0019 |

## TALK (n=11831, 클래스 ['ask_user', 'plan_task', 'respond_only', 'web_search'])
최빈 찍기 정확도: 0.438
| 피처 | 정규화 MI |
|---|---|
| last2_path | 0.1865 |
| prompt_len | 0.1645 |
| last_action | 0.1532 |
| turn | 0.1361 |
| step | 0.1361 |
| hist_len | 0.1352 |
| last_group | 0.1285 |
| open_n | 0.0890 |
| last_result | 0.0694 |
| filler:지금 | 0.0594 |
| elapsed_b | 0.0455 |
| mix | 0.0346 |
| ci | 0.0261 |
| dirty | 0.0218 |
| budget_b | 0.0058 |

## 하드 룰 후보 (조건부 정밀도 ≥99%, 지지도 ≥50)
| 그룹 | 피처 | 값 | 라벨 | 적중/전체 |
|---|---|---|---|---|
| MODIFY | last2_path | write_file|run_bash | edit_file | 138/138 |
| MODIFY | last2_path | glob_pattern|write_file | edit_file | 55/55 |
| EXEC | mix | yaml | run_bash | 340/340 |
