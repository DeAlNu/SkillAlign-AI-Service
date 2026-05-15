"""
Role Configuration — SkillAlign AI.

File ini adalah single source of truth untuk:
  - ROLE_DISPLAY   : mapping role_key → nama tampilan yang manusiawi
  - ROLE_SKILL_MAP : mapping role_key → set skill kategori yang relevan

Digunakan oleh:
  - cv_profile_extractor.py  (untuk tampilkan nama role dari CV)
  - job_title_suggester.py   (untuk hitung skill overlap per role)

Cara menambah role baru:
  1. Tambahkan entry di ROLE_DISPLAY
  2. Tambahkan entry di ROLE_SKILL_MAP dengan set skill kategorinya
  3. Pastikan key di ROLE_PATTERNS (hybrid_scorer.py) konsisten
"""

from typing import Dict, Set
# =============================================================================
# ROLE_DISPLAY
# Mapping dari internal role key → nama tampilan yang manusiawi.
# Key harus konsisten dengan ROLE_PATTERNS di hybrid_scorer.py
# =============================================================================

ROLE_DISPLAY: Dict[str, str] = {
    # Tech — Data & AI
    "data_scientist":    "Data Scientist",
    "data_engineer":     "Data Engineer",
    "data_analyst":      "Data Analyst",
    "ml_engineer":       "Machine Learning Engineer",

    # Tech — Software Engineering
    "frontend":          "Frontend Developer",
    "backend":           "Backend Developer",
    "full_stack":        "Full Stack Developer",
    "mobile":            "Mobile Developer",
    "software_engineer": "Software Engineer",
    "database_administrator": "Database Administrator",

    # Tech — Ops / Design / PM / IT service / IT Consulting
    "devops":            "DevOps Engineer",
    "designer":          "UI/UX Designer",
    "product_manager":   "Product Manager",
    "it_consultant":     "IT Consultant",
    "system_administrator": "System Administrator",
    "cloud_architect": "Cloud Architect",

    # Tech - Security
    "cybersecurity_analyst": "Cybersecurity Analyst",
    "qa_engineer": "QA Engineer",
    "it_support": "IT Support",
    "ai_research_scientist": "AI Research Scientist",

    # Business & Retail
    "project_manager":   "Project Manager",
    "marketing":         "Marketing Specialist",
    "sales":             "Sales Executive",
    "hr_recruiting":     "HR Recruiter",
    "finance":           "Finance Professional",
    "operations":        "Operations Manager",
    "retail":            "Retail Manager",
    "manager_general":   "Manager",
    "supply_chain_manager": "Supply Chain Manager",
    "customer_success_manager": "Customer Success Manager",
    "compliance_officer": "Compliance Officer",
    "financial_analyst": "Financial Analyst",
    "management_consultant": "Management Consultant",
    "business_analyst": "Business Analyst",
    "retail_buyer": "Retail Buyer",
    "store_manager": "Store Manager",

    # Healthcare & hospital & Pharmaceutical Manufacturing
    "nurse":             "Registered Nurse",
    "doctor":            "Physician",
    "healthcare_administrator": "Healthcare Administrator",
    "pharmacist": "Pharmacist",
    "medical_technician": "Medical Technician",
    "clinical_research_coordinator": "Clinical Research Coordinator",
    "hotel_operations_manager": "Hotel Operations Manager",
    "event_coordinator": "Event Coordinator",
    "regulatory_affairs_specialist": "Regulatory Affairs Specialist",
    "pharma_qa_qc_specialist": "QA/QC Specialist",

    # Engineering (non-software) & Civil Engineering
    "civil_eng":         "Civil Engineer",
    "mechanical_eng":    "Mechanical Engineer",
    "electrical_eng":    "Electrical Engineer",
    "architect_design":  "Architect / Urban Planner",
    "construction_pm":   "Construction Project Manager",
    "environmental_engineer": "Environmental Engineer",
    "structural_engineer": "Structural Engineer",
    "transportation_engineer": "Transportation Engineer",
    "construction_estimator": "Construction Estimator",
    "site_safety_manager": "Site Safety Manager",

    # Creative & Advertising Service
    "instructional_designer": "Instructional Designer",
    "content_creator": "Content Creator",
    "technical_writer": "Technical Writer",
    "media_buyer": "Media Buyer",
    "creative_director": "Creative Director",

    # Staffing & Recruiting
    "recruiter": "Recruiter",
    "hr_recruiting": "HR Recruiting",
    "compensation_benefits_specialist": "Compensation & Benefits Specialist",

    # Financial Services & Banking
    "investment_banker": "Investment Banker",
    "credit_risk_analyst": "Credit Risk Analyst",
    "loan_officer": "Loan Officer",
    "treasury_analyst": "Treasury Analyst",
    
    # Insurance
    "underwriter": "Underwriter",
    "claims_adjuster": "Claims Adjuster",
}


# =============================================================================
# ROLE_SKILL_MAP
# Mapping dari internal role key → set skill kategori yang relevan.
# Skill kategori harus konsisten dengan SKILL_DICT di hybrid_scorer.py
# =============================================================================

ROLE_SKILL_MAP: Dict[str, Set[str]] = {
    # ------------------------------------------------------------------
    # Tech — Data & AI
    # ------------------------------------------------------------------
    "data_scientist": {
        "python", "pandas", "numpy", "tensorflow", "pytorch", "sklearn", "machine_learning_algorithms", "nlp", "statistics", "data_visualization", "feature_engineering", "spark", "sql_general", "postgresql", "mongodb","mlops", "model_deployment", "a/b_testing", "git", "jupyter"
    },
    "data_engineer": {
        "python", "pyspark", "spark", "kafka", "airflow", "etl", "sql_general","postgresql", "mysql", "mongodb", "aws", "gcp", "docker", "distributed_systems","data_warehousing", "dbt", "hadoop", "databricks", "terraform", "ci_cd", "git", "data_lake", "parquet", "avro"
    },
    "data_analyst": {
        "sql_general", "python", "pandas", "numpy", "tableau", "powerbi","excel", "statistics", "business_intelligence", "postgresql", "mysql","data_cleaning", "data_visualization", "looker", "google_analytics","critical_thinking", "report_generation", "storytelling_with_data", "git"
    },
    "ml_engineer": {
        "python", "tensorflow", "pytorch", "sklearn", "machine_learning_algorithms", "mlops", "docker", "kubernetes", "aws", "gcp", "ci_cd_ml", "model_serving", "fastapi", "kubeflow", "mlflow", "dvc", "onnx", "terraform", "git", "linux","feature_store"
    },
    "ai_research_scientist": {
        "deep_learning", "nlp", "computer_vision", "pytorch", "tensorflow", "research_publication", "python", "statistics", "mlops", "llm","transformers", "huggingface", "scikit_learn", "pandas", "numpy","model_evaluation", "data_preprocessing", "distributed_training", "git", "linux_cli"
    },

    # ------------------------------------------------------------------
    # Tech — Software Engineering
    # ------------------------------------------------------------------
    "frontend": {
        "javascript", "typescript", "html_css","react", "vue", "angular", "svelte", "next_js", "nuxt_js", "redux", "zustand", "webpack", "vite", "tailwind_css","graphql", "rest_api", "http_protocols", "jest", "cypress", "react_testing_library"
    },
    "backend": {
        "python", "java", "go_lang", "php", "csharp", "ruby", "node_js","spring_boot", "django", "fastapi", "flask", "express", "nestjs", "laravel", "asp_net_core", "gin_gonic","postgresql", "mysql", "mongodb", "redis", "sqlalchemy", "hibernate","rest_api", "graphql", "microservices", "api_design","authentication_oauth", "jwt","docker", "linux_basics", "git"
    },
    "full_stack": {
        "javascript", "typescript", "html_css", "react", "vue", "angular","python", "java", "node_js", "express", "postgresql", "mysql", "mongodb","rest_api", "graphql", "docker", "git", "aws_basics","mern_stack", "mean_stack", "jamstack"
    },
    "mobile": {
        "react_native", "flutter", "dart", "kotlin_multiplatform","swift", "ios_sdk", "xcode", "objective_c","kotlin", "java", "android_sdk", "android_studio","mobile_ui_ux", "app_store_optimization", "push_notifications","rest_api", "json", "git"
    },
    "software_engineer": {
        "python", "java", "javascript", "typescript", "go_lang", "rust", "csharp", "cpp","oop", "data_structures", "algorithms", "system_design_basics","design_patterns", "clean_code", "sql_general", "postgresql", "mysql", "mongodb", "git", "docker", "agile", "scrum", "ci_cd_basics", "unit_testing"
    },
    "database_administrator": {
        "postgresql", "mysql", "oracle", "sql_server", "mongodb", "cassandra","sql_optimization", "query_tuning", "indexing_strategy","backup_recovery", "disaster_recovery", "replication", "sharding","data_security", "access_control", "monitoring_tools", "prometheus","grafana", "pgadmin", "enterprise_manager"
    },
    # ------------------------------------------------------------------
    # Tech — Ops / Design / PM / IT service / IT Consulting
    # ------------------------------------------------------------------
    "devops": {
        "docker", "kubernetes", "terraform", "jenkins", "ansible", "linux","aws", "gcp", "azure", "prometheus", "grafana", "git", "gitlab_ci","argo_cd", "helm", "bash", "python", "infrastructure_as_code","sre", "observability"
    },
    "designer": {
        "figma", "sketch", "adobe_xd", "photoshop", "illustrator", "prototyping", "ui_design", "ux_research", "responsive_design","design_system"
    },
    "product_manager": {
        "agile", "stakeholder_mgmt", "product_strategy", "user_stories","roadmap_planning", "jira", "confluence", "data_analysis", "kpi_tracking","market_research", "prioritization_frameworks", "product_design"
    },
    "it_consultant": {
        "it_strategy", "business_analysis", "requirement_analysis", "solution_building", "stakeholder_communication", "python","sql_general", "agile", "cloud_migration", "gap_analysis"
    },
    "system_administrator": {
        "linux", "windows_server", "vmware", "active_directory", "hyper_v", "network_admin", "dns", "dhcp", "powershell", "bash", "backup_recovery","itil", "monitoring_tools", "security_basics"
    },
    "cloud_architect": {
        "aws", "gcp", "azure", "terraform", "solution_design", "cost_optimization", "security_architecture", "landing_zone", "serverless", "microservices_architecture", "disaster_recovery", "compliance_basics"
    },

    # ------------------------------------------------------------------
    # Tech - Security
    # ------------------------------------------------------------------
    "cybersecurity_analyst": {
        "network_security", "vulnerability_assessment", "siem", "incident_response", "compliance_frameworks", "firewalls", "linux", "python", "security_monitoring", "threat_intelligence", "edr", "ids_ips", "splunk", "wireshark", "nmap", "risk_assessment"
    },
    "qa_engineer": {
        "automation_testing", "selenium", "pytest", "api_testing", "postman", "jira", "manual_testing", "agile", "ci_cd_testing", "jenkins", "testng", "junit", "git", "sql_basics", "performance_testing", "test_plan", "bug_tracking", "cypress"
    },
    "it_support": {
        "troubleshooting", "helpdesk_support", "active_directory", "windows_os","ticketing_systems", "customer_service", "macos", "remote_desktop", "vpn_support", "printer_setup", "hardware_troubleshooting", "office365","google_workspace", "basic_networking", "mdm", "antivirus_support", "documentation"
    },

    # ------------------------------------------------------------------
    # Business & Retail
    # ------------------------------------------------------------------
    "project_manager": {
        "pmp", "agile", "scrum", "kanban", "stakeholder_mgmt", "team_leadership",
        "jira", "confluence", "microsoft_project", "risk_management", "budgeting",
        "project_scheduling", "cross_functional_collaboration", "reporting"
    },
    "marketing": {
        "seo", "sem", "google_ads", "meta_ads", "social_media", "content_marketing",
        "email_marketing", "marketing_automation", "crm_marketing", "copywriting",
        "marketing_analytics", "brand_strategy", "digital_marketing", "campaign_mgmt",
        "roi_optimization", "advertising", "google_analytics", "a/b_testing"
    },
    "sales": {
        "b2b_sales", "b2c_sales", "crm", "salesforce", "hubspot", "lead_generation",
        "negotiation", "sales_forecasting", "pipeline_management", "account_planning",
        "stakeholder_mgmt", "regional_mgmt", "kpi_tracking", "cold_calling",
        "presentation_skills"
    },
    "finance": {
        "accounting", "auditing", "financial_modeling", "ifrs_gaap", "tax",
        "risk_mgmt", "underwriting", "claims_processing", "investment_analysis",
        "excel_advanced", "vba", "power_bi", "tableau", "sql_general",
        "sap_finance", "valuation", "corporate_finance", "treasury", "internal_controls"
    },
    "operations": {
        "lean_six_sigma", "team_leadership", "stakeholder_mgmt", "production_planning",
        "supply_chain_basics", "process_improvement", "kpi_management",
        "inventory_control", "quality_management", "continuous_improvement",
        "erp", "data_analysis", "budget_management"
    },
    "retail": {
        "retail_ops", "store_mgmt", "customer_service", "staff_training",
        "pos_systems", "inventory_management", "visual_merchandising",
        "sales_analysis", "shrinkage_control", "loss_prevention",
        "staff_scheduling", "omni_channel_retail", "customer_loyalty_programs"
    },
    "supply_chain_manager": {
        "procurement", "logistics", "inventory_mgmt", "erp", "sap", "lean_six_sigma",
        "vendor_mgmt", "forecasting", "demand_planning", "warehouse_management",
        "transportation_mgmt", "supply_chain_optimization", "global_sourcing",
        "import_export", "sap_scm", "oracle_scm", "power_bi", "excel_advanced"
    },
    "customer_success_manager": {
        "crm", "client_onboarding", "retention_strategy", "account_management",
        "communication", "data_analysis", "saas_metrics", "customer_lifetime_value",
        "nps", "churn_reduction", "upselling", "cross_selling", "customer_feedback",
        "zendesk", "hubspot", "salesforce_crm", "quarterly_business_review",
        "customer_health_score"
    },
    "compliance_officer": {
        "regulatory_compliance", "risk_assessment", "gdpr", "audit",
        "legal_research", "policy_development", "hipaa_compliance", "soc2",
        "iso_27001", "sox_compliance", "aml", "kyc", "internal_controls",
        "compliance_monitoring", "ethics_training", "regulatory_filing", "breach_response"
    },
    "financial_analyst": {
        "financial_modeling", "excel", "excel_power_query", "vba", "bloomberg_terminal",
        "valuation", "accounting", "sql_general", "data_analysis", "risk_mgmt",
        "power_bi", "tableau", "forecasting", "budgeting", "variance_analysis",
        "quickbooks", "sap_finance", "presentation_skills"
    },
    "management_consultant": {
        "business_strategy", "market_research", "process_improvement",
        "financial_modeling", "stakeholder_analysis", "presentation_design",
        "change_management", "excel_advanced", "power_bi", "tableau",
        "problem_solving", "hypothesis_driven", "client_relationship",
        "competitive_analysis", "operational_excellence", "workshop_facilitation"
    },
    "business_analyst": {
        "requirements_gathering", "process_mapping", "data_analysis", "sql_general",
        "excel", "agile", "scrum", "stakeholder_mgmt", "jira", "confluence",
        "user_stories", "acceptance_criteria", "brd", "gap_analysis",
        "visio", "bpmn", "uml", "user_acceptance_testing"
    },
    "retail_buyer": {
        "merchandising_strategy", "vendor_negotiation", "inventory_forecasting",
        "trend_analysis", "excel", "supply_chain_basics", "purchasing",
        "assortment_planning", "vendor_management", "open_to_buy", "cost_negotiation",
        "margin_analysis", "po_management", "erp_retail", "retail_math",
        "consumer_behavior", "market_research"
    },
    "store_manager": {
        "store_ops", "staff_scheduling", "loss_prevention", "customer_service",
        "pos_systems", "inventory_control", "sales_reporting", "staff_training",
        "performance_reviews", "team_leadership", "budget_management",
        "visual_merchandising", "local_marketing", "kpi_tracking",
        "inventory_accuracy", "cash_management", "hr_compliance", "health_safety"
    },
    # ------------------------------------------------------------------
    # Healthcare & hospital & Pharmaceutical Manufacturing
    # ------------------------------------------------------------------
    "nurse": {
        "clinical", "rn_credential", "patient_assessment", "care_planning",
        "medication_administration", "medical_records", "hipaa_compliance",
        "emr_ehr", "pharmacy", "life_support", "wound_care", "vital_signs_monitoring",
        "team_collaboration", "emergency_response", "infection_control", "patient_education"
    },
    "doctor": {
        "clinical", "patient_diagnosis", "treatment_planning", "differential_diagnosis",
        "medical_records", "hipaa_compliance", "emr_ehr", "prescribing_medications",
        "specialty_knowledge", "medical_procedures", "patient_consultation",
        "multidisciplinary_collaboration", "evidence_based_practice"
    },
    "healthcare_administrator": {
        "healthcare_operations", "ehr_systems", "medical_billing", "staff_scheduling",
        "compliance", "patient_relations", "revenue_cycle_management",
        "cms_regulations", "budget_management", "quality_improvement",
        "accreditation", "hospital_administration", "policy_development",
        "data_analysis_healthcare", "patient_satisfaction_metrics"
    },
    "pharmacist": {
        "pharmacology", "drug_dispensing", "drug_interactions", "patient_counseling",
        "regulatory_compliance", "inventory_mgmt", "medication_therapy_management",
        "pharmacy_software", "vaccination_administration", "prescription_verification",
        "compounding", "dea_compliance", "medication_safety", "clinical_pharmacy"
    },
    "medical_technician": {
        "phlebotomy", "medical_imaging", "lab_testing", "patient_care",
        "hipaa_compliance", "emr_ehr", "clinical_protocols", "specimen_processing",
        "quality_control_lab", "equipment_maintenance", "ecg_ekg",
        "sterilization_techniques", "infection_prevention", "point_of_care_testing",
        "laboratory_information_system", "venipuncture"
    },
    "clinical_research_coordinator": {
        "clinical_trials", "gcp", "regulatory_compliance", "data_collection",
        "patient_recruitment", "medical_records", "project_management", "informed_consent",
        "protocol_adherence", "site_management", "adverse_event_reporting",
        "source_document_verification", "irb_submissions", "ctms",
        "patient_follow_up", "data_entry_accuracy"
    },
    "hotel_operations_manager": {
        "hospitality_management", "revenue_management", "guest_relations",
        "staff_scheduling", "budgeting", "pos_systems", "crs_pms",
        "front_office_ops", "housekeeping_management", "fandb_management",
        "guest_satisfaction_scores", "online_reputation_management",
        "yield_management", "inventory_management_hotel", "health_safety_hotel",
        "staff_training_hotel"
    },
    "event_coordinator": {
        "event_planning", "vendor_management", "budget_tracking", "logistics",
        "client_relations", "risk_management", "marketing_basics", "venue_selection",
        "contract_negotiation", "timeline_management", "on_site_coordination",
        "av_management", "registration_management", "post_event_evaluation",
        "sponsorship_management", "event_marketing", "guest_list_management"
    },
    "regulatory_affairs_specialist": {
        "gmp", "fda_regulations", "regulatory_submissions", "compliance",
        "quality_assurance", "documentation", "clinical_trial_support", "ctd",
        "ind_nda_anda", "ema_regulations", "ich_guidelines", "regulatory_strategy",
        "post_market_surveillance", "labeling_compliance", "ectd",
        "health_authority_interactions"
    },
    "pharma_qa_qc_specialist": {
        "quality_assurance", "lab_testing", "process_validation", "gmp",
        "data_integrity", "audit_readiness", "sop_development", "hplc", "gc",
        "stability_studies", "batch_release", "deviation_management", "change_control",
        "capa", "quality_metrics", "cleaning_validation", "sterility_testing"
    },
    # ------------------------------------------------------------------
    # Engineering (non-software) & Civil Engineering
    # ------------------------------------------------------------------
    "civil_eng": {
        "autocad", "civil_3d", "revit", "construction_pm", "structural_analysis",
        "site_development", "grading_design", "stormwater_management",
        "earthwork_estimation", "building_codes", "permit_acquisition", "quantity_takeoff"
    },
    "mechanical_eng": {
        "autocad", "solidworks", "catia", "creo", "ansys", "matlab",
        "finite_element_analysis", "computational_fluid_dynamics", "thermal_analysis",
        "product_design", "mfg_eng", "hvac", "piping_design", "materials_selection",
        "tolerance_analysis", "prototyping", "mech_pe", "root_cause_analysis"
    },
    "electrical_eng": {
        "autocad", "matlab", "circuit_design", "pcb_layout", "power_systems",
        "plc_programming", "scada", "renewable_energy", "motor_control",
        "instrumentation", "ieee_standards", "electrical_safety", "schematics_reading",
        "oscilloscope", "multimeter"
    },
    "architect_design": {
        "autocad", "revit", "sketchup", "rhino_3d", "lumion", "photoshop_architecture",
        "building_information_modeling", "spatial_planning", "construction_drawings",
        "3d_visualization", "building_codes", "material_specifications", "permit_drawings"
    },
    "construction_pm": {
        "pmp", "construction_management", "project_scheduling", "primavera_p6",
        "budget_control", "subcontractor_management", "risk_management_construction",
        "quality_control_construction", "osha_safety", "contract_administration",
        "rfis_and_submittals", "change_order_management", "site_logistics", "agile"
    },
    "environmental_engineer": {
        "environmental_assessment", "sustainability", "waste_management", "gis",
        "regulatory_compliance", "autocad", "life_cycle_analysis", "environmental_impact_statement",
        "water_treatment", "air_quality_monitoring", "remediation_design", "epa_regulations",
        "hazmat_handling", "environmental_sampling", "site_assessment", "environmental_modeling"
    },
    "structural_engineer": {
        "structural_analysis", "autocad", "revit", "staad_pro", "sap2000", "etabs",
        "matlab", "concrete_design", "steel_design", "foundation_design",
        "building_codes", "seismic_design", "wind_load_calculation", "load_combinations",
        "retaining_walls", "structural_drawings", "quantity_estimation", "site_inspection"
    },
    "transportation_engineer": {
        "traffic_analysis", "transportation_planning", "autocad", "gis",
        "infrastructure_design", "feasibility_study", "safety_standards",
        "hcs", "synchro", "vissim", "microsimulation", "roadway_design",
        "pavement_design", "signal_timing", "traffic_impact_study",
        "multimodal_planning", "aashto_standards"
    },
    "construction_estimator": {
        "construction_estimating", "blueprint_reading", "cost_analysis", "quantity_takeoff",
        "contract_administration", "excel", "scheduling", "procurement", "bidding_process",
        "unit_cost_pricing", "historical_data_analysis", "estimating_software",
        "bid_package_preparation", "value_engineering", "subcontractor_quotes"
    },
    "site_safety_manager": {
        "osha_compliance", "osha_30", "site_inspection", "risk_assessment",
        "hazard_identification", "job_safety_analysis", "safety_training",
        "incident_reporting", "stakeholder_mgmt", "construction_safety",
        "ppe_management", "emergency_response_plan", "confined_space_safety",
        "fall_protection", "hot_work_permit", "lockout_tagout", "safety_audit",
        "subcontractor_safety_supervision"
    },
    # ------------------------------------------------------------------
    # Creative & Advertising Service
    # ------------------------------------------------------------------
    "instructional_designer": {
        "e_learning", "lms", "curriculum_development", "adult_learning_theory",
        "multimedia_design", "assessment_design", "articulate_360", "captivate",
        "camtasia", "storyboarding", "scorm_xapi", "learning_objectives",
        "gamification", "accessibility", "addie", "content_migration",
        "collaboration_sme", "learning_analytics"
    },
    "content_creator": {
        "copywriting", "video_editing", "social_media_strategy", "seo",
        "graphic_design", "analytics", "brand_voice", "content_calendar",
        "canva", "adobe_premiere_pro", "photoshop", "illustrator",
        "instagram_meta", "tiktok_strategy", "youtube_seo", "content_management_systems",
        "photo_editing", "hashtag_strategy", "engagement_metrics", "user_generated_content"
    },
    "technical_writer": {
        "technical_writing", "documentation", "api_documentation", "content_strategy",
        "editing", "information_architecture", "style_guides", "xml", "dita",
        "markdown", "docs_as_code", "madcap_flare", "github_pages", "readthedocs",
        "snagit", "adobe_framemaker", "jira_confluence", "user_manual",
        "release_notes", "quick_start_guides", "communication", "research",
        "software_development_basics", "content_management_systems"
    },
    "media_buyer": {
        "media_planning", "programmatic_advertising", "campaign_optimization",
        "analytics", "budget_mgmt", "negotiation", "roi_tracking", "google_ads",
        "meta_ads_manager", "dv360", "the_trade_desk", "amazon_ads", "tiktok_ads",
        "cpm_cpc_cpa", "audience_targeting", "pacing_strategy", "ad_serving",
        "attribution_modeling", "forecasting", "competitive_analysis"
    },
    "creative_director": {
        "creative_direction", "art_direction", "brand_strategy", "adobe_creative_suite",
        "copywriting", "client_presentation", "team_leadership", "campaign_concepting",
        "visual_identity", "design_thinking", "storytelling", "feedback_provision",
        "cross_channel_campaigns", "budget_management_creative", "vendor_management",
        "quality_assurance_creative", "trend_forecasting", "portfolio_management"
    },
    # ------------------------------------------------------------------
    # Staffing & Recruiting
    # ------------------------------------------------------------------
    "recruiter": {
        "candidate_sourcing", "boolean_search", "linkedin_recruiter", "interviewing",
        "candidate_screening", "ats", "employer_branding", "talent_acquisition",
        "labor_law", "onboarding", "offer_negotiation", "background_checks",
        "reference_checking", "recruitment_marketing", "diversity_metrics",
        "recruitment_reporting", "interview_coordination", "candidate_relationship_management"
    },
    "hr_recruiting": {
        "talent_acquisition", "staffing", "team_leadership", "recruitment_strategy",
        "workforce_planning", "sourcing_strategies", "ats_expert", "recruitment_metrics",
        "employer_value_proposition", "university_relations", "executive_search",
        "recruitment_budgeting", "vendor_management", "hr_compliance_recruitment",
        "employer_branding_campaigns", "reporting_analytics_recruitment"
    },
    "compensation_benefits_specialist": {
        "compensation_analysis", "benefits_admin", "payroll", "excel",
        "hr_metrics", "labor_law", "stakeholder_mgmt", "salary_benchmarking",
        "job_evaluation", "market_pricing", "incentive_design", "total_rewards",
        "hris_compensation", "benefits_administration", "open_enrollment",
        "compliance_compensation", "hr_audit", "vlookup_pivot_excel",
        "data_visualization_compensation", "communication_employee_benefits"
    },
    # ------------------------------------------------------------------
    # Financial Services & Banking
    # ------------------------------------------------------------------
    "investment_banker": {
        "financial_modeling", "mergers_acquisitions", "valuation", "dcf_analysis",
        "comparable_company_analysis", "precedent_transactions", "leveraged_buyout",
        "excel", "pitch_books", "deal_execution", "due_diligence", "client_advisory",
        "capital_raising", "ipo_process", "sector_analysis", "regulatory_compliance",
        "bloomberg_terminal", "presentation_skills_investment"
    },
    "credit_risk_analyst": {
        "credit_analysis", "risk_modeling", "financial_statements", "credit_scoring",
        "probability_of_default", "loss_given_default", "exposure_at_default",
        "sql_general", "python", "sas", "r_programming", "regulatory_reporting",
        "stress_testing", "ifrs9", "cecl", "basel_iii", "risk_rating_systems",
        "portfolio_risk_analysis", "power_bi_risk", "credit_memo_writing", "collateral_analysis"
    },
    "loan_officer": {
        "loan_processing", "credit_analysis", "underwriting_loans", "customer_relations",
        "financial_statements", "debt_to_income_ratio", "ltv_calculation",
        "loan_origination", "mortgage_lending", "commercial_lending", "consumer_lending",
        "loan_documentation", "closing_process", "regulatory_compliance", "federal_regulations",
        "fair_lending", "banking_software", "encompass", "crm_lending", "risk_assessment",
        "sales_skills_lending", "portfolio_management_loans"
    },
    "treasury_analyst": {
        "liquidity_management", "cash_flow_forecasting", "financial_modeling",
        "excel", "sql_general", "risk_mgmt", "banking_operations", "cash_conversion_cycle",
        "working_capital_management", "debt_covenant_management", "investment_portfolio_treasury",
        "foreign_exposure", "interest_rate_risk", "treasury_management_systems",
        "bank_relationship_management", "payment_processing", "reconciliation_treasury",
        "short_term_investments", "commercial_paper", "line_of_credit_management",
        "power_bi_treasury", "excel_power_query"
    },
    # ------------------------------------------------------------------
    # Insurance
    # ------------------------------------------------------------------
    "underwriter": {
        "policy_analysis", "financial_modeling", "regulatory_compliance", "data_analysis",
        "risk_assessment_underwriting", "pricing_strategy", "loss_ratio_analysis",
        "exposure_analysis", "reinsurance_basics", "underwriting_guidelines",
        "quote_generation", "policy_issuance", "renewal_analysis", "submission_evaluation",
        "industry_knowledge", "appetite_alignment", "portfolio_management_uw",
        "underwriting_software", "excel_advanced_uw", "communication_uw",
        "stakeholder_collaboration"
    },
    "claims_adjuster": {
        "claims_processing", "investigation_skills", "negotiation", "customer_service",
        "documentation", "fraud_detection", "insurance_software", "damage_assessment",
        "reserve_setting", "coverage_analysis", "liability_determination",
        "settlement_negotiation", "claim_closure", "subrogation", "litigation_support",
        "field_inspection", "estimating_software", "medical_claims_review",
        "compliance_claims", "claim_investigation", "vendor_management_claims",
        "reporting_claims"
    },
}
