-- 1. Login as SYS:-
-- sqlplus / as sysdba

-- 2. Switch to Correct PDB:-
ALTER SESSION SET CONTAINER = FREEPDB1;

-- 3. Create User (Only if NOT exists):-
-- Run this only once
CREATE USER bank_modern IDENTIFIED BY bank;

GRANT CREATE SESSION, CREATE TABLE, CREATE VIEW, CREATE SEQUENCE TO bank_modern;

ALTER USER bank_modern QUOTA UNLIMITED ON USERS;
ALTER USER bank_modern DEFAULT TABLESPACE USERS;

-- 4. Connect as User:-
CONNECT bank_modern/bank@localhost:1521/FREEPDB1

-- 5. Create Tables (NO schema prefix needed)
CREATE TABLE customers (
    id NUMBER GENERATED AS IDENTITY PRIMARY KEY,
    first_name VARCHAR2(100),
    last_name VARCHAR2(100),
    email VARCHAR2(255)
);

CREATE TABLE accounts (
    id NUMBER GENERATED AS IDENTITY PRIMARY KEY,
    customer_id NUMBER,
    balance NUMBER(10,2) DEFAULT 0,
    CONSTRAINT fk_customer FOREIGN KEY (customer_id)
    REFERENCES customers(id)
);

CREATE TABLE transactions (
    id NUMBER GENERATED AS IDENTITY PRIMARY KEY,
    account_id NUMBER,
    txn_type VARCHAR2(20),
    amount NUMBER(10,2),
    created_at TIMESTAMP DEFAULT SYSTIMESTAMP,
    CONSTRAINT fk_account FOREIGN KEY (account_id)
    REFERENCES accounts(id)
);

-- 6. Create Index:-
CREATE INDEX idx_transactions_account_created 
ON transactions(account_id, created_at);

-- 7. Verify Tables:-
SELECT * FROM customers;
SELECT * FROM accounts;
SELECT * FROM transactions;

-- SELECT * FROM all_tables WHERE owner = 'BANK_MODERN';