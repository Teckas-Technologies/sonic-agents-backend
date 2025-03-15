from web3 import Web3
from app.config import INFURA_API_KEY

INFURA_URL = f"https://mainnet.infura.io/v3/{INFURA_API_KEY}"
SEPOLIA_TESTNET_URL = f"https://sepolia.infura.io/v3/{INFURA_API_KEY}"
web3 = Web3(Web3.HTTPProvider(SEPOLIA_TESTNET_URL))

def get_transaction_summary(tx_hash: str):
    """Fetches transaction details and returns a summary."""
    try:
        receipt = web3.eth.get_transaction_receipt(tx_hash)
        if not receipt:
            return {"status": "pending", "message": "⏳ Transaction is still pending."}

        status = "✅ Confirmed" if receipt.status == 1 else "❌ Failed"
        return {
            "transactionHash": tx_hash,
            "status": status,
            "gasUsed": receipt.gasUsed,
            "blockNumber": receipt.blockNumber
        }
    except Exception as e:
        return {"error": str(e)}
    
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
]
    
def read_from_contract():
    WALLET_ADDRESS="0xFf43E33C40276FEEff426C5448cF3AD9df6b5741"
    USDT_CONTRACT_ADDRESS = "0xD382e755E3162827D4e1E554af08b0855F961E01"

    if web3.is_connected():
        print("Connected to Sepolia Network")
        
        # Get ETH Balance
        eth_balance = web3.eth.get_balance(WALLET_ADDRESS)
        eth_balance = web3.from_wei(eth_balance, "ether")
        print(f"ETH Balance: {eth_balance} ETH")

        # Load USDT Contract
        usdt_contract = web3.eth.contract(address=USDT_CONTRACT_ADDRESS, abi=ERC20_ABI)

        # Get USDT Balance
        usdt_balance = usdt_contract.functions.balanceOf(WALLET_ADDRESS).call()
        
        # Get Token Decimals
        usdt_decimals = usdt_contract.functions.decimals().call()
        usdt_balance = usdt_balance / (10 ** usdt_decimals)

        print(f"USDT Balance: {usdt_balance} USDT")

    else:
        print("Failed to connect to the Sepolia Network")