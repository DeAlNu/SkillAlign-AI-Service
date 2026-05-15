Payload /api/v1/skill-gap test case 1
```json
{
    "cv_text": "Experienced Senior Software Engineer with 6+ years of expertise in building scalable web applications using Python, Django, and React.js. Proven track record in designing RESTful APIs, optimizing PostgreSQL databases, and implementing CI/CD pipelines with Docker and Kubernetes. Strong background in cloud infrastructure on AWS (EC2, S3, Lambda). Skilled in agile methodologies, cross-functional team collaboration, and mentoring junior developers. Passionate about clean code, test-driven development, and solving complex technical challenges.",
    "job_description": "Senior Backend Engineer: We are looking for an experienced Python developer to build high-performance microservices. Requirements include 5+ years of experience with Django or FastAPI, strong knowledge of SQL databases, and experience with AWS cloud services. You will be responsible for API design, system architecture, and ensuring scalability."
}
```

Payload /api/v1/skill-gap test case 2
```json
{
  "cv_text": "Experienced Senior Software Engineer with 6+ years of expertise in building scalable web applications using Python, Django, and React.js. Proven track record in designing RESTful APIs, optimizing PostgreSQL databases, and implementing CI/CD pipelines with Docker and Kubernetes. Strong background in cloud infrastructure on AWS (EC2, S3, Lambda). Skilled in agile methodologies, cross-functional team collaboration, and mentoring junior developers. Passionate about clean code, test-driven development, and solving complex technical challenges.",
    "job_description": "Full Stack Engineer: Seeking a versatile developer comfortable with both backend (Python/Node.js) and frontend (React/Vue) technologies. You will work on end-to-end feature development, from database schema design to UI implementation. 4+ years of commercial experience preferred."
}
```

Payload /api/v1/skill-gap test case 2
```json
curl -X POST "https://api.example.com/api/v1/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
  "cv_text": "Senior Project Manager with 8+ years of experience leading cross-functional teams in diverse sectors including IT, Healthcare, Construction, and Manufacturing. PMP certified professional with a strong track record in budget management, risk mitigation, and agile methodologies. Skilled in stakeholder communication, process improvement, and delivering complex projects on time. Proven ability to adapt to dynamic environments and drive operational efficiency.",
  "job_descriptions": [
    "Talent Acquisition Specialist (Staffing and Recruiting): We are seeking a proactive recruiter to source and hire top-tier talent for our clients. Must have experience with full-cycle recruiting, applicant tracking systems, and building talent pipelines.",
    "Hospital Operations Manager (Hospitals and Health Care): Oversee daily administrative and clinical operations of the facility. Requires knowledge of healthcare regulations, staff scheduling, and budget management to ensure high-quality patient care.",
    "IT Project Manager (IT Services and IT Consulting): Lead the delivery of IT consulting projects for enterprise clients. Must be familiar with SDLC, cloud migration strategies, and managing technical teams.",
    "Financial Analyst (Financial Services): Analyze financial data and market trends to provide investment recommendations. Strong proficiency in Excel and financial modeling is required.",
    "Construction Site Manager (Construction): Supervise construction sites to ensure projects are built according to plans and safety regulations. Experience with residential and commercial building projects is preferred.",
    "Scrum Master (Software Development): Facilitate agile ceremonies and remove impediments for the development team. Must have a deep understanding of Scrum framework and Jira administration.",
    "Production Supervisor (Manufacturing): Manage production line operations to meet quality and quantity targets. Experience with Lean Manufacturing and Six Sigma methodologies is a plus.",
    "Management Consultant (Business Consulting and Services): Advise clients on strategic business initiatives and operational improvements. Requires strong analytical skills and ability to present findings to C-level executives.",
    "Retail Store Manager (Retail): Responsible for the overall performance of the store, including sales targets, inventory management, and customer service standards.",
    "Insurance Underwriter (Insurance): Evaluate insurance applications to determine coverage terms and premiums. Must have strong attention to detail and knowledge of risk assessment principles.",
    "Civil Engineer (Civil Engineering): Design and oversee infrastructure projects such as roads, bridges, and water systems. PE license is preferred.",
    "Branch Manager (Banking): Lead branch operations, manage staff, and ensure compliance with banking regulations while driving business growth.",
    "Hotel General Manager (Hospitality): Oversee all aspects of hotel operations including front desk, housekeeping, and food & beverage to ensure guest satisfaction.",
    "Account Executive (Advertising Services): Manage client relationships and lead the development of advertising campaigns. Strong presentation and negotiation skills are essential.",
    "Quality Assurance Specialist (Pharmaceutical Manufacturing): Ensure manufacturing processes comply with GMP and FDA regulations. Experience in pharmaceutical quality control is required."
  ],
  "user_id": "candidate_pm_001"
}'
```