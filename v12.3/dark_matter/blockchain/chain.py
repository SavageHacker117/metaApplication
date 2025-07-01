
import hashlib
import json
from time import time
from typing import List, Dict, Any, Optional
from uuid import uuid4

from ..data_contracts.dm_schemas import Block, Transaction, BlockchainStatus

class Blockchain:
    """
    A lightweight, permissioned blockchain for the Dark Matter meta-environmental layer.
    Implements Proof-of-Authority (PoA) consensus for state integrity and auditability.
    """

    def __init__(self):
        self.chain: List[Block] = []
        self.current_transactions: List[Transaction] = []
        self.nodes: set = set() # For a permissioned blockchain, these would be pre-approved
        self.audit_logs: List[Dict[str, Any]] = []

        # Create the genesis block
        self.new_block(proof=100, previous_hash=\'1\', validator=\'genesis_node\')
        self.add_audit_log(\'system\', \'Genesis block created\', {})

    def new_block(self, proof: int, previous_hash: Optional[str] = None, validator: str = \'default_validator\') -> Block:
        """
        Creates a new Block and adds it to the chain.
        :param proof: The proof given by the Proof of Authority algorithm
        :param previous_hash: Hash of previous Block
        :param validator: Identifier of the validator node that created this block
        :return: New Block
        """
        block = Block(
            index=len(self.chain) + 1,
            timestamp=time(),
            transactions=self.current_transactions,
            proof=proof,
            previous_hash=previous_hash or self.hash_block(self.chain[-1]) if self.chain else \'1\',
            hash=\'\'
        )
        block.hash = self.hash_block(block)
        self.current_transactions = []
        self.chain.append(block)
        self.add_audit_log(\'system\', f\'Block {block.index} created by {validator}\' , block.dict())
        return block

    def new_transaction(self, sender: str, recipient: str, payload: Dict[str, Any], signature: str) -> str:
        """
        Creates a new transaction to go into the next mined Block
        :param sender: Address of the Sender
        :param recipient: Address of the Recipient
        :param payload: Data payload of the transaction (e.g., environment config, metadata)
        :param signature: Digital signature of the transaction
        :return: The index of the Block that will hold this transaction
        """
        transaction = Transaction(
            timestamp=time(),
            sender=sender,
            recipient=recipient,
            payload=payload,
            signature=signature,
            transaction_id=str(uuid4())
        )
        self.current_transactions.append(transaction)
        self.add_audit_log(sender, f\'Transaction added: {transaction.transaction_id}\' , transaction.dict())
        return self.last_block.index + 1

    @property
    def last_block(self) -> Block:
        """
        Returns the last Block in the chain.
        """
        return self.chain[-1]

    @staticmethod
    def hash_block(block: Block) -> str:
        """
        Creates a SHA-256 hash of a Block.
        :param block: Block
        :return: Hash string
        """
        # We must make sure that the Dictionary is Ordered, or we'll have inconsistent hashes
        block_string = json.dumps(block.dict(), sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def is_chain_valid(self, chain: List[Block]) -> bool:
        """
        Determines if a given blockchain is valid.
        :param chain: A blockchain
        :return: True if valid, False if not
        """
        if not chain: # Handle empty chain case
            return True

        last_block = chain[0]
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]
            # Check that the hash of the previous block is correct
            if block.previous_hash != self.hash_block(last_block):
                return False

            # Check that the Proof of Authority is correct (simplified for now)
            # In a real PoA, this would involve checking validator signatures and stake
            # For this lightweight version, we just ensure proof is positive
            if block.proof <= 0:
                return False

            last_block = block
            current_index += 1

        return True

    def get_status(self) -> BlockchainStatus:
        """
        Returns the current status of the blockchain.
        """
        return BlockchainStatus(
            chain_length=len(self.chain),
            latest_block_hash=self.last_block.hash,
            pending_transactions=len(self.current_transactions),
            nodes=list(self.nodes),
            consensus_status=\'Proof-of-Authority (Simplified)\'
        )

    def add_audit_log(self, actor: str, action: str, details: Dict[str, Any]):
        """
        Adds an entry to the internal audit log.
        """
        log_entry = {
            "timestamp": time(),
            "actor": actor,
            "action": action,
            "details": details
        }
        self.audit_logs.append(log_entry)

    def get_audit_logs(self) -> List[Dict[str, Any]]:
        """
        Retrieves all audit logs.
        """
        return self.audit_logs

    def rollback_chain(self, target_block_index: int) -> bool:
        """
        Rolls back the blockchain to a specified block index.
        This is a critical operation and should be used with extreme caution.
        :param target_block_index: The index of the block to roll back to.
        :return: True if rollback successful, False otherwise.
        """
        if not (1 <= target_block_index < len(self.chain)):
            print(f"Invalid target block index {target_block_index}. Chain length is {len(self.chain)}.")
            return False

        # Keep transactions from rolled back blocks as pending
        for i in range(target_block_index, len(self.chain)):
            self.current_transactions.extend(self.chain[i].transactions)

        self.chain = self.chain[:target_block_index]
        self.add_audit_log(\'system\', f\'Blockchain rolled back to block {target_block_index}\' , {\'new_chain_length\': len(self.chain)})
        print(f"Blockchain successfully rolled back to block {target_block_index}.")
        return True




