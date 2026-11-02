# Consolidated benchmark table — v3

Aggregated from `outputs/results_registry.json` (mean ± std over seeds).
Rows are keyed by task/model/variant **and test-set size**, so results on
different splits never mix. `provenance`: `current` (matches live data
fingerprint), `legacy` (pre-provenance registry entry), `mixed`.

## ablation
| model | variant | n_seeds | test_n | n_pos | provenance | best_val_pr_auc | epochs | test_ew_before_frac | test_pr_auc | test_rca_jaccard | test_roc_auc | test_type_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tagn | full | 1 | MISSING | MISSING | legacy | 0.6118 | 9.0000 | 0.5167 | 0.5531 | 0.6504 | 0.8840 | 0.2287 |
| tagn | no_graph | 1 | MISSING | MISSING | legacy | 0.9516 | 11.0000 | 0.4333 | 0.5216 | 0.6928 | 0.8338 | 0.3767 |
| tagn | static_graph | 1 | MISSING | MISSING | legacy | 0.8507 | 11.0000 | 0.3500 | 0.8320 | 0.8813 | 0.9525 | 0.6951 |
| tagn | undirected | 1 | MISSING | MISSING | legacy | 0.5712 | 11.0000 | 0.4333 | 0.5685 | 0.7901 | 0.8963 | 0.5516 |
| tagn | no_nodeconv | 1 | MISSING | MISSING | legacy | 0.4514 | 9.0000 | 0.4833 | 0.5096 | 0.6023 | 0.8373 | 0.2152 |
| tagn | hazard_only | 1 | MISSING | MISSING | legacy | 0.0430 | 8.0000 | 1.0000 | 0.0354 | 0.5513 | 0.3614 | 0.0404 |

## anomaly_detection
| model | variant | n_seeds | test_n | n_pos | provenance | f1_max | f1_max_test_optimal_optimistic | pr_auc | precision | recall | roc_auc | threshold | threshold_val_tuned | tpr_at_fpr_0.01 | tpr_at_fpr_0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp_009_hybrid | w=0.5 | 5 | 4969 | 424.0000 ± 0.0000 | current | 0.9023 ± 0.0039 | MISSING | 0.9092 ± 0.0028 | 0.9430 ± 0.0057 | 0.8651 ± 0.0061 | 0.9405 ± 0.0010 | 0.3109 ± 0.1480 | 0.3109 ± 0.1480 | 0.8788 ± 0.0046 | 0.8976 ± 0.0043 |
| exp_009_reference_logreg | w=0.5 | 5 | 4969 | 424.0000 ± 0.0000 | current | 0.8804 ± 0.0000 | MISSING | 0.8946 ± 0.0000 | 0.8932 ± 0.0000 | 0.8679 ± 0.0000 | 0.9290 ± 0.0000 | 0.5624 ± 0.0000 | 0.5624 ± 0.0000 | 0.8679 ± 0.0000 | 0.8821 ± 0.0000 |
| exp_009_tagn | w=0.5 | 5 | 4969 | 424.0000 ± 0.0000 | current | 0.8986 ± 0.0111 | MISSING | 0.9236 ± 0.0039 | 0.9148 ± 0.0205 | 0.8830 ± 0.0039 | 0.9554 ± 0.0025 | -0.3379 ± 0.2268 | -0.3379 ± 0.2268 | 0.8873 ± 0.0065 | 0.9127 ± 0.0071 |
| hybrid_seed_ensemble |  | 1 | 4768 | 223.0000 | legacy | 0.9522 | MISSING | 0.9704 | 0.9676 | 0.9372 | 0.9885 | 1.0414 | MISSING | 0.9686 | 0.9776 |
| sup_logreg |  | 1 | 4969 | 424.0000 | current | 0.8638 | 0.8659 | 0.8717 | 0.9597 | 0.7854 | 0.9114 | 0.8397 | 0.8397 | 0.8231 | 0.8679 |
| sup_xgb |  | 1 | 4969 | 424.0000 | current | 0.7043 | 0.7147 | 0.7317 | 0.8449 | 0.6038 | 0.8953 | 0.0876 | 0.0876 | 0.6038 | 0.6816 |
| tagn |  | 3 | 4768 | 223.0000 ± 0.0000 | legacy | 0.5750 ± 0.0829 | MISSING | 0.6077 ± 0.1148 | 0.6508 ± 0.0908 | 0.5157 ± 0.0786 | 0.8865 ± 0.0600 | -1.2574 ± 0.9398 | -1.2574 ± 0.9398 | 0.5067 ± 0.0875 | 0.6069 ± 0.1537 |
| tagn |  | 5 | 4969 | 424.0000 ± 0.0000 | legacy | 0.8986 ± 0.0111 | MISSING | 0.9236 ± 0.0039 | 0.9148 ± 0.0205 | 0.8830 ± 0.0039 | 0.9554 ± 0.0025 | -0.3379 ± 0.2268 | -0.3379 ± 0.2268 | 0.8873 ± 0.0065 | 0.9127 ± 0.0071 |
| tagn_seed_ensemble |  | 1 | 4768 | 223.0000 | legacy | 0.9500 | MISSING | 0.9728 | 0.9631 | 0.9372 | 0.9920 | 0.5400 | MISSING | 0.9641 | 0.9776 |
| unsup_isolation_forest |  | 1 | 4969 | 424.0000 | current | 0.5152 | 0.5218 | 0.3869 | 0.4405 | 0.6203 | 0.8505 | 0.4868 | 0.4868 | 0.1250 | 0.4269 |
| unsup_lof |  | 1 | 4969 | 424.0000 | current | 0.7711 | 0.7775 | 0.8167 | 0.8015 | 0.7429 | 0.9462 | 3.1614 | 3.1614 | 0.6580 | 0.8302 |
| unsup_one_class_svm |  | 1 | 4969 | 424.0000 | current | 0.7027 | 0.7170 | 0.6839 | 0.7734 | 0.6439 | 0.8418 | -296.2261 | -296.2261 | 0.6156 | 0.7052 |

## cross_vendor_detection
| model | variant | n_seeds | test_n | n_pos | provenance | f1_max | pr_auc | precision | recall | roc_auc | threshold | tpr_at_fpr_0.01 | tpr_at_fpr_0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gaussian_zs |  | 1 | 475 | 185.0000 | legacy | 0.5606 | 0.3328 | 0.3895 | 1.0000 | 0.2808 | -50509696018107170816.0000 | 0.0649 | 0.0649 |
| logreg_transfer_loo |  | 1 | 475 | 185.0000 | legacy | 0.9179 | 0.8920 | 0.8732 | 0.9676 | 0.9441 | -2.1949 | 0.1243 | 0.7784 |
| logreg_zs |  | 1 | 475 | 185.0000 | legacy | 0.7535 | 0.6232 | 0.7727 | 0.7351 | 0.7049 | -544529.5468 | 0.1027 | 0.1297 |
| naive_z |  | 1 | 475 | 185.0000 | legacy | 0.8928 | 0.8365 | 0.8287 | 0.9676 | 0.9251 | 0.5845 | 0.1243 | 0.3946 |

## deferral_rca
| model | variant | n_seeds | test_n | n_pos | provenance | best_jaccard | rcai.stability_mean |
|---|---|---|---|---|---|---|---|
| tagn_baseline |  | 1 | MISSING | MISSING | legacy | 0.3191 | 0.9965 |
| tagn_grl |  | 1 | MISSING | MISSING | legacy | MISSING | 0.9992 |

## early_warning
| model | variant | n_seeds | test_n | n_pos | provenance | mean_abs_lead_err_steps | median_lag_steps | median_lead_steps | n_chunks | p10_lead_steps | p90_lead_steps | per_step_ap_mean | per_step_auc_mean | warning_before_onset_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tagn |  | 9 | MISSING | MISSING | legacy | 48.3612 ± 12.1366 | -23.3750 ± 25.5060 | -13.0000 | 78.3333 ± 17.3925 | -97.5000 | 45.9000 | 0.6266 | 0.4529 | 0.6473 ± 0.1876 |

## explainability
| model | variant | n_seeds | test_n | n_pos | provenance | attn_template_vs_non_auc | jaccard_topk_vs_template_mean |
|---|---|---|---|---|---|---|---|
| tagn |  | 1 | MISSING | MISSING | legacy | 0.8519 | 0.6449 |

## loro_generalization
| model | variant | n_seeds | test_n | n_pos | provenance | anomaly_pr_auc | anomaly_roc_auc | rca_topk_shift_jaccard | rca_type_template_jaccard | type_accuracy | type_macro_f1_seen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tagn |  | 1 | MISSING | MISSING | legacy | 0.7588 | 0.9030 | 0.7950 | 0.7690 | 0.3757 | 0.3385 |

## robustness
| model | variant | n_seeds | test_n | n_pos | provenance | f1_max | pr_auc | precision | recall | roc_auc | threshold | tpr_at_fpr_0.01 | tpr_at_fpr_0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tagn |  | 1 | 4768 | 223.0000 | legacy | 0.5511 | 0.5042 | 0.8900 | 0.3991 | 0.8250 | 0.5002 | 0.4126 | 0.4888 |

## root_cause
| model | variant | n_seeds | test_n | n_pos | provenance | topk_shift_jaccard | type_template_jaccard |
|---|---|---|---|---|---|---|---|
| tagn |  | 8 | MISSING | MISSING | legacy | 0.4608 ± 0.1702 | 0.5411 ± 0.1551 |

## root_cause_baselines
| model | variant | n_seeds | test_n | n_pos | provenance | input_shift_topK.jaccard_mean | input_shift_topK.micro_f1 | recording_prior.jaccard_mean | recording_prior.micro_f1 | type_prior_ml.jaccard_mean | type_prior_ml.micro_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rca |  | 1 | MISSING | MISSING | current | 0.3398 | 0.5213 | 0.6025 | 0.7135 | 0.5533 | 0.7020 |

## type_classification
| model | variant | n_seeds | test_n | n_pos | provenance | accuracy | macro_f1_seen | micro_f1 | test_types_present | test_types_total | type_accuracy | type_macro_f1_seen | type_macro_recall_over_seen | type_micro_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tagn |  | 8 | MISSING | MISSING | legacy | 0.2719 ± 0.1436 | 0.2036 ± 0.0537 | 0.2719 ± 0.1436 | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| xgb |  | 1 | MISSING | MISSING | current | MISSING | MISSING | MISSING | 11.0000 | 11.0000 | 0.2193 | 0.1988 | 0.2907 | 0.2193 |

## type_invariant_rca
| model | variant | n_seeds | test_n | n_pos | provenance | best_jaccard | tcrl.macro_f1 | tcrl.majority | tcrl.n | tcrl.probe_acc | tcrl.tcrl | test.f1_max | test.n | test.n_positives | test.pr_auc | test.precision | test.recall | test.roc_auc | test.threshold | test.threshold_val_tuned | test.tpr_at_fpr_0.01 | test.tpr_at_fpr_0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tagn_baseline |  | 1 | MISSING | MISSING | legacy | MISSING | 0.6761 | 0.4126 | 223.0000 | 0.7964 | 0.6535 | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| tagn_baseline_v2 |  | 1 | MISSING | MISSING | legacy | 0.2955 | 0.6773 | 0.4126 | 223.0000 | 0.8036 | 0.6656 | 0.9502 | 4768.0000 | 223.0000 | 0.9689 | 0.9589 | 0.9417 | 0.9918 | 0.8062 | 0.8062 | 0.9507 | 0.9821 |
| tagn_grl |  | 1 | MISSING | MISSING | legacy | MISSING | 0.8471 | 0.2170 | 424.0000 | 0.8868 | 0.8554 | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| tagn_grl_v2 |  | 1 | MISSING | MISSING | legacy | 0.6589 | 0.7871 | 0.4126 | 223.0000 | 0.8304 | 0.7112 | 0.9443 | 4768.0000 | 223.0000 | 0.9673 | 0.9381 | 0.9507 | 0.9927 | -0.0408 | -0.0408 | 0.9552 | 0.9821 |
