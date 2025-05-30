"""
Schema information and metadata for the RAG-Enhanced SQL Query Generator.
Contains table definitions, join conditions, and T-SQL rules for the insurance database.
"""

from typing import List, Dict, Any

# Schema Information as Table Chunks
TABLE_CHUNKS: List[Dict[str, Any]] = [
    {
        "type": "table",
        "table": "dwh.dim_claims",
        "text": "Table: dwh.dim_claims | Columns: claim_reference_id, date_claim_first_notified, date_of_loss_from, date_claim_opened, date_of_loss_to, cause_of_loss_code, loss_description, date_coverage_confirmed, date_closed, date_claim_amount_agreed, date_paid_final_amount, date_fees_paid_final_amount, date_reopened, date_claim_denied, date_claim_withdrawn, status, refer_to_underwriters, denial_indicator, reason_for_denial, claim_total_claimed_amount, settlement_currency_code, indemnity_amount_paid, fees_amount_paid, expenses_paid_amount, dw_ins_upd_dt, org_id",
        "description": "Contains claim metadata and lifecycle events including claim status, dates, amounts, and denial information.",
        "columns": [
            "claim_reference_id", "date_claim_first_notified", "date_of_loss_from", "date_claim_opened", 
            "date_of_loss_to", "cause_of_loss_code", "loss_description", "date_coverage_confirmed", 
            "date_closed", "date_claim_amount_agreed", "date_paid_final_amount", "date_fees_paid_final_amount", 
            "date_reopened", "date_claim_denied", "date_claim_withdrawn", "status", "refer_to_underwriters", 
            "denial_indicator", "reason_for_denial", "claim_total_claimed_amount", "settlement_currency_code", 
            "indemnity_amount_paid", "fees_amount_paid", "expenses_paid_amount", "dw_ins_upd_dt", "org_id"
        ],
        "keywords": ["claims", "claim", "loss", "damage", "settlement", "denial", "status", "indemnity", "fees"]
    },
    {
        "type": "table",
        "table": "dwh.dim_policy",
        "text": "Table: dwh.dim_policy | Columns: Id, agreement_id, policy_number, new_or_renewal, group_reference, broker_reference, changed_date, effective_date, start_date_time, expiry_date_time, renewal_date_time, product_code, product_name, country_code, country, a3_country_code, country_sub_division_code, class_of_business_code, classof_business_name, main_line_of_business_name, insurance_type, section_details_number, section_details_code, section_details_name, line_of_business, section_details_description, dw_ins_upd_dt, org_id, document_id",
        "description": "Stores policy details, effective dates, product information, and geographical coverage.",
        "columns": [
            "Id", "agreement_id", "policy_number", "new_or_renewal", "group_reference", "broker_reference", 
            "changed_date", "effective_date", "start_date_time", "expiry_date_time", "renewal_date_time", 
            "product_code", "product_name", "country_code", "country", "a3_country_code", 
            "country_sub_division_code", "class_of_business_code", "classof_business_name", 
            "main_line_of_business_name", "insurance_type", "section_details_number", "section_details_code", 
            "section_details_name", "line_of_business", "section_details_description", "dw_ins_upd_dt", 
            "org_id", "document_id"
        ],
        "keywords": ["policy", "policies", "coverage", "product", "business", "renewal", "effective", "expiry"]
    },
    {
        "type": "table",
        "table": "dwh.fact_claims_dtl",
        "text": "Table: dwh.fact_claims_dtl | Columns: Id, claim_reference_id, agreement_id, policy_number, org_id, riskitems_id, Payment_Detail_Settlement_Currency_Code, Paid_Amount, Expenses_Paid_Total_Amount, Coverage_Legal_Fees_Total_Paid_Amount, Defence_Legal_Fees_Total_Paid_Amount, Adjusters_Fees_Total_Paid_Amount, TPAFees_Paid_Amount, Fees_Paid_Amount, Incurred_Detail_Settlement_Currency_Code, Indemnity_Amount, Expenses_Amount, Coverage_Legal_Fees_Amount, Defence_Fees_Amount, Adjuster_Fees_Amount, TPAFees_Amount, Fees_Amount, indemnity_reserves_amount, dw_ins_upd_dt, indemnity_amount_paid",
        "description": "Detailed claim financial information including payments, expenses, fees, and reserves.",
        "columns": [
            "Id", "claim_reference_id", "agreement_id", "policy_number", "org_id", "riskitems_id", 
            "Payment_Detail_Settlement_Currency_Code", "Paid_Amount", "Expenses_Paid_Total_Amount", 
            "Coverage_Legal_Fees_Total_Paid_Amount", "Defence_Legal_Fees_Total_Paid_Amount", 
            "Adjusters_Fees_Total_Paid_Amount", "TPAFees_Paid_Amount", "Fees_Paid_Amount", 
            "Incurred_Detail_Settlement_Currency_Code", "Indemnity_Amount", "Expenses_Amount", 
            "Coverage_Legal_Fees_Amount", "Defence_Fees_Amount", "Adjuster_Fees_Amount", 
            "TPAFees_Amount", "Fees_Amount", "indemnity_reserves_amount", "dw_ins_upd_dt", 
            "indemnity_amount_paid"
        ],
        "keywords": ["claim details", "payments", "expenses", "fees", "legal", "adjusters", "reserves", "financial"]
    },
    {
        "type": "table",
        "table": "dwh.fact_premium",
        "text": "Table: dwh.fact_premium | Columns: Id, agreement_id, policy_number, org_id, riskitems_id, original_currency_code, total_paid, instalments_amount, taxes_amount_paid, commission_percentage, commission_amount_paid, brokerage_amount_paid, insurance_amount_paid, additional_fees_paid, settlement_currency_code, gross_premium_settlement_currency, brokerage_amount_paid_settlement_currency, net_premium_settlement_currency, commission_amount_paid_settlement_currency, final_net_premium_settlement_currency, rate_of_exchange, total_settlement_amount_paid, date_paid, transaction_type, net_amount, gross_premium_paid_this_time, final_net_premium, tax_amount, dw_ins_upd_dt",
        "description": "Premium payment transactions including commissions, taxes, brokerage, and currency information.",
        "columns": [
            "Id", "agreement_id", "policy_number", "org_id", "riskitems_id", "original_currency_code", 
            "total_paid", "instalments_amount", "taxes_amount_paid", "commission_percentage", 
            "commission_amount_paid", "brokerage_amount_paid", "insurance_amount_paid", 
            "additional_fees_paid", "settlement_currency_code", "gross_premium_settlement_currency", 
            "brokerage_amount_paid_settlement_currency", "net_premium_settlement_currency", 
            "commission_amount_paid_settlement_currency", "final_net_premium_settlement_currency", 
            "rate_of_exchange", "total_settlement_amount_paid", "date_paid", "transaction_type", 
            "net_amount", "gross_premium_paid_this_time", "final_net_premium", "tax_amount", 
            "dw_ins_upd_dt"
        ],
        "keywords": ["premium", "payments", "commission", "brokerage", "taxes", "instalments", "currency"]
    },
    {
        "type": "table",
        "table": "dwh.fct_policy",
        "text": "Table: dwh.fct_policy | Columns: Id, agreement_id, policy_number, org_id, start_date, annual_premium, sum_insured, limit_of_liability, final_net_premium, tax_amount, final_net_premium_settlement_currency, settlement_currency_code, gross_premium_before_taxes_amount, dw_ins_upd_dt, document_id, gross_premium_paid_this_time",
        "description": "Policy summary information including premiums, limits, and financial aggregates.",
        "columns": [
            "Id", "agreement_id", "policy_number", "org_id", "start_date", "annual_premium", 
            "sum_insured", "limit_of_liability", "final_net_premium", "tax_amount", 
            "final_net_premium_settlement_currency", "settlement_currency_code", 
            "gross_premium_before_taxes_amount", "dw_ins_upd_dt", "document_id", 
            "gross_premium_paid_this_time"
        ],
        "keywords": ["policy summary", "annual premium", "sum insured", "liability", "limits", "aggregates"]
    }
]

# Join Conditions and Table Usage Guidelines
JOIN_CONDITIONS: str = """
Join Conditions:
- fact_claims_dtl.claim_reference_id = dim_claims.claim_reference_id AND fact_claims_dtl.org_id = dim_claims.org_id
- fct_policy.policy_number = dim_policy.policy_number AND fct_policy.org_id = dim_policy.org_id
- fact_premium.policy_number = dim_policy.policy_number AND fact_premium.org_id = dim_policy.org_id

Table Usage Guidelines:
- Use `dwh.fact_premium` for premium/payment-related metrics and transactions
- Use `dwh.dim_claims` or `dwh.fact_claims_dtl` for claim-related details and financials
- Use `dwh.dim_policy` for policy metadata (start/end dates, renewals, products)
- Use `dwh.fct_policy` for policy-level financial summaries

Date Field Guidelines:
- Use `date_paid` for premium payment dates in fact_premium
- Use `date_claim_opened`, `date_closed`, etc. for claims in dim_claims
- Use `effective_date`, `expiry_date_time` for policies in dim_policy
"""

# T-SQL Rules and Requirements
TSQL_RULES: str = """
T-SQL Rules and Requirements:
- Use proper T-SQL syntax only (no PostgreSQL, MySQL, or other SQL dialects)
- Use DATEPART(), YEAR(), MONTH(), DAY() for date functions instead of EXTRACT()
- Use ISNULL() instead of COALESCE() when possible
- Use TOP instead of LIMIT
- For pagination, use OFFSET...FETCH NEXT instead of LIMIT
- Use proper JOIN syntax with explicit INNER/LEFT/RIGHT/FULL OUTER
- Do NOT use the column alias in the GROUP BY or ORDER BY clauses.
- Instead, repeat the full expression used in the SELECT clause inside GROUP BY and ORDER BY
- For string operations, use LEN() instead of LENGTH(), CHARINDEX() instead of POSITION()
- Use GETDATE() for current datetime, not NOW()
- For conditional logic, prefer CASE WHEN over IIF() for compatibility
- Do NOT use NULLS FIRST/NULLS LAST in ORDER BY (not supported in T-SQL)
- Use proper table aliases and qualify column names where ambiguous
- For date formatting, use FORMAT() or CONVERT() functions
- Use appropriate data types: VARCHAR(MAX), NVARCHAR(MAX), DECIMAL, DATETIME2, etc.
- Date time format example: 2024-11-21 06:57:57.000
"""

# Utility Functions for Schema Operations
def get_table_by_name(table_name: str) -> Dict[str, Any]:
    """Get table information by table name."""
    for table in TABLE_CHUNKS:
        if table["table"] == table_name:
            return table
    return {}

def get_tables_by_keyword(keyword: str) -> List[Dict[str, Any]]:
    """Get tables that contain a specific keyword."""
    matching_tables = []
    keyword_lower = keyword.lower()
    
    for table in TABLE_CHUNKS:
        # Check if keyword is in table keywords
        if any(kw.lower() == keyword_lower for kw in table["keywords"]):
            matching_tables.append(table)
        # Check if keyword is in table name
        elif keyword_lower in table["table"].lower():
            matching_tables.append(table)
        # Check if keyword is in description
        elif keyword_lower in table["description"].lower():
            matching_tables.append(table)
    
    return matching_tables

def get_all_table_names() -> List[str]:
    """Get all table names from the schema."""
    return [table["table"] for table in TABLE_CHUNKS]

def get_all_keywords() -> List[str]:
    """Get all unique keywords from all tables."""
    all_keywords = []
    for table in TABLE_CHUNKS:
        all_keywords.extend(table["keywords"])
    return list(set(all_keywords))

def get_table_columns(table_name: str) -> List[str]:
    """Get columns for a specific table."""
    table_info = get_table_by_name(table_name)
    return table_info.get("columns", [])

def validate_table_exists(table_name: str) -> bool:
    """Validate if a table exists in the schema."""
    return any(table["table"] == table_name for table in TABLE_CHUNKS)

def get_related_tables(primary_table: str) -> List[str]:
    """Get tables that can be joined with the primary table based on join conditions."""
    related_tables = []
    
    # Define join relationships
    join_relationships = {
        "dwh.dim_claims": ["dwh.fact_claims_dtl"],
        "dwh.fact_claims_dtl": ["dwh.dim_claims"],
        "dwh.dim_policy": ["dwh.fct_policy", "dwh.fact_premium"],
        "dwh.fct_policy": ["dwh.dim_policy"],
        "dwh.fact_premium": ["dwh.dim_policy"]
    }
    
    return join_relationships.get(primary_table, [])

def construct_schema_prompt(relevant_tables: List[Dict[str, Any]]) -> str:
    """Construct a schema section for prompts using relevant tables."""
    schema_section = ""
    for table_info in relevant_tables:
        schema_section += f"""
TABLE: {table_info['table']}
Description: {table_info['description']}
Columns: {', '.join(table_info['columns'])}
"""
    return schema_section

def get_schema_summary() -> Dict[str, Any]:
    """Get a summary of the entire schema."""
    return {
        "total_tables": len(TABLE_CHUNKS),
        "table_names": get_all_table_names(),
        "total_columns": sum(len(table["columns"]) for table in TABLE_CHUNKS),
        "all_keywords": get_all_keywords(),
        "fact_tables": [table["table"] for table in TABLE_CHUNKS if "fact" in table["table"]],
        "dimension_tables": [table["table"] for table in TABLE_CHUNKS if "dim" in table["table"]]
    }

# Export all schema components
__all__ = [
    "TABLE_CHUNKS",
    "JOIN_CONDITIONS", 
    "TSQL_RULES",
    "get_table_by_name",
    "get_tables_by_keyword",
    "get_all_table_names",
    "get_all_keywords",
    "get_table_columns",
    "validate_table_exists",
    "get_related_tables",
    "construct_schema_prompt",
    "get_schema_summary"
]