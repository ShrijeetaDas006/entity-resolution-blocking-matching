from normalize import normalize_name, normalize_address

print(normalize_name("Acme Corp"))
print(normalize_name("ACME Corporation"))
print(normalize_address("123 Main St, Near SBI ATM"))