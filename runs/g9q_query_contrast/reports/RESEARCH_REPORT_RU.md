# FlyGraph G9Q

{
  "run": "G9Q",
  "status": "completed",
  "hypothesis": "hard same-state query negatives improve query grounding over random negatives at matched scoring compute",
  "parameters_each": 469648,
  "new_random_initializations": 0,
  "ancestral_seeds": [
    7601,
    7602,
    7603
  ],
  "completed_continuations": 6,
  "steps_each": 1000,
  "metrics": {
    "hard_query_negative": {
      "iid": {
        "question_accuracy_mean": 0.13749999999999998,
        "pair_accuracy_mean": 0.0,
        "query_sensitivity_mean": 0.016666666666666666,
        "correct_by_seed": [
          50,
          52,
          63
        ],
        "both_correct_by_seed": [
          0,
          0,
          0
        ]
      },
      "surface": {
        "question_accuracy_mean": 0.13333333333333333,
        "pair_accuracy_mean": 0.0,
        "query_sensitivity_mean": 0.024999999999999998,
        "correct_by_seed": [
          49,
          41,
          70
        ],
        "both_correct_by_seed": [
          0,
          0,
          0
        ]
      },
      "extrapolation": {
        "question_accuracy_mean": 0.0,
        "pair_accuracy_mean": 0.0,
        "query_sensitivity_mean": 0.008333333333333333,
        "correct_by_seed": [
          0,
          0,
          0
        ],
        "both_correct_by_seed": [
          0,
          0,
          0
        ]
      },
      "counterfactual": {
        "question_accuracy_mean": 0.12083333333333333,
        "pair_accuracy_mean": 0.0,
        "query_sensitivity_mean": 0.024999999999999998,
        "correct_by_seed": [
          45,
          40,
          60
        ],
        "both_correct_by_seed": [
          0,
          0,
          0
        ]
      }
    },
    "random_negative_control": {
      "iid": {
        "question_accuracy_mean": 0.13916666666666666,
        "pair_accuracy_mean": 0.0,
        "query_sensitivity_mean": 0.013333333333333334,
        "correct_by_seed": [
          52,
          60,
          55
        ],
        "both_correct_by_seed": [
          0,
          0,
          0
        ]
      },
      "surface": {
        "question_accuracy_mean": 0.115,
        "pair_accuracy_mean": 0.0,
        "query_sensitivity_mean": 0.03,
        "correct_by_seed": [
          39,
          43,
          56
        ],
        "both_correct_by_seed": [
          0,
          0,
          0
        ]
      },
      "extrapolation": {
        "question_accuracy_mean": 0.010833333333333334,
        "pair_accuracy_mean": 0.0,
        "query_sensitivity_mean": 0.02,
        "correct_by_seed": [
          0,
          0,
          13
        ],
        "both_correct_by_seed": [
          0,
          0,
          0
        ]
      },
      "counterfactual": {
        "question_accuracy_mean": 0.12,
        "pair_accuracy_mean": 0.0,
        "query_sensitivity_mean": 0.018333333333333337,
        "correct_by_seed": [
          47,
          41,
          56
        ],
        "both_correct_by_seed": [
          0,
          0,
          0
        ]
      }
    }
  },
  "gains_pp": {
    "iid": {
      "question_accuracy_mean": -0.16666666666666774,
      "pair_accuracy_mean": 0.0,
      "query_sensitivity_mean": 0.3333333333333332
    },
    "surface": {
      "question_accuracy_mean": 1.8333333333333326,
      "pair_accuracy_mean": 0.0,
      "query_sensitivity_mean": -0.5000000000000001
    },
    "extrapolation": {
      "question_accuracy_mean": -1.0833333333333335,
      "pair_accuracy_mean": 0.0,
      "query_sensitivity_mean": -1.1666666666666667
    },
    "counterfactual": {
      "question_accuracy_mean": 0.08333333333333387,
      "pair_accuracy_mean": 0.0,
      "query_sensitivity_mean": 0.6666666666666661
    }
  },
  "paired": {
    "7601": {
      "iid": {
        "wins": 12,
        "losses": 14,
        "delta_pp": -0.5,
        "exact_two_sided_p": 0.8450189828872681
      },
      "surface": {
        "wins": 24,
        "losses": 14,
        "delta_pp": 2.5,
        "exact_two_sided_p": 0.14330665429588407
      },
      "extrapolation": {
        "wins": 0,
        "losses": 0,
        "delta_pp": 0.0,
        "exact_two_sided_p": 1.0
      },
      "counterfactual": {
        "wins": 12,
        "losses": 14,
        "delta_pp": -0.5,
        "exact_two_sided_p": 0.8450189828872681
      }
    },
    "7602": {
      "iid": {
        "wins": 14,
        "losses": 22,
        "delta_pp": -2.0,
        "exact_two_sided_p": 0.2429849540349096
      },
      "surface": {
        "wins": 18,
        "losses": 20,
        "delta_pp": -0.5,
        "exact_two_sided_p": 0.8714146793645341
      },
      "extrapolation": {
        "wins": 0,
        "losses": 0,
        "delta_pp": 0.0,
        "exact_two_sided_p": 1.0
      },
      "counterfactual": {
        "wins": 20,
        "losses": 21,
        "delta_pp": -0.25,
        "exact_two_sided_p": 1.0
      }
    },
    "7603": {
      "iid": {
        "wins": 23,
        "losses": 15,
        "delta_pp": 2.0,
        "exact_two_sided_p": 0.2558750795433298
      },
      "surface": {
        "wins": 28,
        "losses": 14,
        "delta_pp": 3.5,
        "exact_two_sided_p": 0.04355852192384191
      },
      "extrapolation": {
        "wins": 0,
        "losses": 13,
        "delta_pp": -3.25,
        "exact_two_sided_p": 0.000244140625
      },
      "counterfactual": {
        "wins": 17,
        "losses": 13,
        "delta_pp": 1.0,
        "exact_two_sided_p": 0.584664711728692
      }
    }
  },
  "gates": {
    "iid_question_gain_ge3pp": false,
    "iid_pair_gain_ge2pp": false,
    "query_sensitivity_gain_ge10pp": false,
    "positive_iid_question_gain_seeds": false,
    "surface_extrap_pair_regression_ok": true
  },
  "verdict": "NOT_CONFIRMED_IN_REGISTERED_PROTOCOL",
  "training_compute_match": {
    "scored_sequences_per_step": 32,
    "updates": 1000,
    "batch_pairs": 8,
    "target_bytes_equal": true
  },
  "scope": "small synthetic memory-grounding probe; not300-800M, not Qwen parity"
}
