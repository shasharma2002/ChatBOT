import re
from typing import Optional
import parsedatetime
from datetime import datetime

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def validate_phone(phone: str) -> bool:
    """Validate phone number format"""
    pattern = r'^\+?1?\d{9,15}$'
    return bool(re.match(pattern, phone))

def parse_date(date_string: str) -> Optional[str]:
    """Parse date string to YYYY-MM-DD format"""
    cal = parsedatetime.Calendar()
    time_struct, parse_status = cal.parse(date_string)
    if parse_status:
        return datetime(*time_struct[:3]).strftime('%Y-%m-%d')
    return None
