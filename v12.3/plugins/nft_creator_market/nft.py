import os, json, hashlib, datetime

class NFTMarketplace:
    def __init__(self, engine):
        self.engine = engine
        self.nfts = []  # List of minted NFTs

    def mint_nft(self, creator, asset_path, meta, open_source=True):
        # Hash asset for uniqueness
        with open(asset_path, "rb") as f:
            asset_bytes = f.read()
        asset_hash = hashlib.sha256(asset_bytes).hexdigest()
        nft = {
            "id": asset_hash,
            "creator": creator,
            "asset_path": asset_path,
            "meta": meta,
            "minted_at": datetime.datetime.utcnow().isoformat(),
            "open_source": open_source,
            "for_sale": False,
            "price": None
        }
        self.nfts.append(nft)
        print(f"[RUBY][NFT] Minted NFT: {asset_hash}, open_source={open_source}")
        # (Optional: Interact with blockchain contract/mint here)
        return nft

    def list_nft(self, nft_id, price):
        for nft in self.nfts:
            if nft["id"] == nft_id:
                nft["for_sale"] = True
                nft["price"] = price
                print(f"[RUBY][NFT] NFT {nft_id} listed for {price}")
                return nft
        print(f"[RUBY][NFT] NFT {nft_id} not found.")
        return None

    def sell_nft(self, nft_id, buyer):
        for nft in self.nfts:
            if nft["id"] == nft_id and nft["for_sale"]:
                nft["owner"] = buyer
                nft["for_sale"] = False
                print(f"[RUBY][NFT] NFT {nft_id} sold to {buyer}")
                return nft
        print(f"[RUBY][NFT] NFT {nft_id} not available for sale.")
        return None

    def marketplace_status(self):
        # Return list of all NFTs for sale
        return [nft for nft in self.nfts if nft["for_sale"]]

def register(pluginAPI):
    m = NFTMarketplace(pluginAPI.engine)
    pluginAPI.provide("mint_nft", m.mint_nft)
    pluginAPI.provide("list_nft", m.list_nft)
    pluginAPI.provide("sell_nft", m.sell_nft)
    pluginAPI.provide("marketplace_status", m.marketplace_status)
    print("[RUBY] nft_creator_market plugin registered.")
