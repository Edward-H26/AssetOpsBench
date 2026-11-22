"""The hydraulic test rig hyd_1: its profile, asset.json, and its record loader, records.py. The package declares the label of
a start and the three tables that the code of the rig computes on first use, by reference, so that it imports nothing."""
# the separability label of the rig with the E18 grade of the CausalDiagnosis server nested, in the form the verifier reads
LABELS = "CausalDiagnosis.environment.reference_policies.diagnosability_label:start_grade"
TABLES = ("CausalDiagnosis.environment.reference_policies.diagnosability_label:LABELS_PATH", "CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.diagnosability_grade:GRADES_PATH",
          "CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models:MODELS_PATH")
