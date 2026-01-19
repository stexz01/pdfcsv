"""
Banking Keywords Configuration
PDFcsv - Universal PDF to CSV Extractor

This file contains keywords used to detect bank statements.
Contributions welcome for additional languages and regional terms.

Usage:
    - If PDF header contains these keywords → bank statement detected
    - User is warned about potential empty debit/credit columns
"""

# Column header keywords that indicate debit/withdrawal
DEBIT_KEYWORDS = {
    # English (Global)
    'debit', 'withdrawal', 'withdraw', 'withdrawn', 'dr', 'dr.',
    'withdrawals', 'debit amt', 'debit amount',
    'money out', 'paid out', 'outflow',

    # Spanish (Spain, Latin America)
    'débito', 'retiro', 'cargo', 'salida',

    # French (France, Canada, Africa)
    'débit', 'retrait', 'sortie',

    # German (Germany, Austria, Switzerland)
    'soll', 'abbuchung', 'auszahlung', 'lastschrift',

    # Portuguese (Brazil, Portugal)
    'débito', 'saque', 'saída',

    # Italian
    'debito', 'prelievo', 'addebito',

    # Dutch
    'debet', 'opname', 'af',

    # Hindi (India)
    'निकासी', 'डेबिट',

    # Arabic (Middle East, North Africa)
    'مدين', 'سحب',

    # Chinese (China, Taiwan, Singapore)
    '借方', '支出', '取款',

    # Japanese
    '引出', '出金', '借方',

    # Korean
    '출금', '인출', '차변',

    # Russian
    'дебет', 'списание', 'расход',

    # Turkish
    'borç', 'çekilen',
}

# Column header keywords that indicate credit/deposit
CREDIT_KEYWORDS = {
    # English (Global)
    'credit', 'deposit', 'deposited', 'cr', 'cr.',
    'deposits', 'credit amt', 'credit amount',
    'money in', 'paid in', 'inflow',

    # Spanish
    'crédito', 'depósito', 'abono', 'entrada',

    # French
    'crédit', 'dépôt', 'entrée', 'versement',

    # German
    'haben', 'einzahlung', 'gutschrift',

    # Portuguese
    'crédito', 'depósito', 'entrada',

    # Italian
    'credito', 'deposito', 'accredito',

    # Dutch
    'credit', 'storting', 'bij',

    # Hindi
    'जमा', 'क्रेडिट',

    # Arabic
    'دائن', 'إيداع',

    # Chinese
    '贷方', '存入', '存款',

    # Japanese
    '預入', '入金', '貸方',

    # Korean
    '입금', '대변',

    # Russian
    'кредит', 'зачисление', 'приход',

    # Turkish
    'alacak', 'yatırılan',
}

# General banking keywords (to detect if it's a bank statement)
BANKING_KEYWORDS = {
    # English
    'balance', 'transaction', 'statement', 'account',
    'particulars', 'description', 'narration', 'reference',
    'cheque', 'check', 'chq', 'ref', 'txn', 'amt',
    'opening balance', 'closing balance', 'available balance',

    # Common abbreviations
    'a/c', 'acc', 'acct', 'bal', 'trn', 'val',

    # Spanish
    'saldo', 'transacción', 'cuenta', 'movimiento',

    # French
    'solde', 'transaction', 'compte', 'opération',

    # German
    'saldo', 'kontostand', 'buchung', 'konto',

    # Portuguese
    'saldo', 'transação', 'conta', 'extrato',

    # Hindi
    'शेष', 'लेनदेन', 'खाता',

    # Arabic
    'رصيد', 'حساب', 'معاملة',

    # Chinese
    '余额', '交易', '账户', '对账单',

    # Japanese
    '残高', '取引', '口座', '明細',
}


def is_bank_statement(text: str) -> bool:
    """Check if text contains banking keywords"""
    text_lower = text.lower()

    # Check for debit/credit keywords
    has_debit = any(kw in text_lower for kw in DEBIT_KEYWORDS)
    has_credit = any(kw in text_lower for kw in CREDIT_KEYWORDS)
    has_banking = any(kw in text_lower for kw in BANKING_KEYWORDS)

    # If has both debit and credit terms, likely a bank statement
    if has_debit and has_credit:
        return True

    # If has banking terms plus either debit or credit
    if has_banking and (has_debit or has_credit):
        return True

    return False


def detect_debit_credit_columns(headers: list) -> dict:
    """
    Detect which columns are debit/credit from header names
    Returns dict with column indices
    """
    result = {
        'debit_col': None,
        'credit_col': None,
        'balance_col': None,
    }

    for i, header in enumerate(headers):
        h_lower = header.lower().strip()

        if any(kw in h_lower for kw in DEBIT_KEYWORDS):
            result['debit_col'] = i
        elif any(kw in h_lower for kw in CREDIT_KEYWORDS):
            result['credit_col'] = i
        elif 'balance' in h_lower or 'saldo' in h_lower or '余额' in h_lower:
            result['balance_col'] = i

    return result
