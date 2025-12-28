import sys

def save_token():
    # 1. Ask you to paste the new long token
    print("Paste your new 'Bearer' token below and hit ENTER:")
    token = input().strip()
    
    # 2. Add 'Bearer ' if you forgot it
    if not token.startswith("Bearer "):
        token = "Bearer " + token
        
    # 3. Save it to a hidden file that your bot reads
    with open("token.txt", "w") as f:
        f.write(token)
    
    print("\n✅ Token updated! Your bot is good for another 24 hours.")

if __name__ == "__main__":
    save_token()