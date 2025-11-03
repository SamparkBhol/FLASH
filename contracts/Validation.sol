// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title IZKPVerifier
 * @dev This is the interface for a separate, pre-deployed ZKP Verifier
 * contract (e.g., a Groth16 verifier). Our main contract will call this.
 * This separation of concerns is standard practice.
 */
interface IZKPVerifier {
    /**
     * @dev Simulates a ZKP verification.
     * @param proof The zero-knowledge proof bytes.
     * @param publicInputs A hash of public inputs (e.g., old and new model fingerprints).
     * @return true if the proof is valid.
     */
    function verify(bytes calldata proof, bytes32 publicInputs)
        external
        view
        returns (bool);
}

/**
 * @title FlashImmuneValidator
 * @author FLASH-IMMUNE Research Team
 * @dev This contract manages node reputation, validates model updates via ZKP
 * and statistical consensus, and quarantines malicious nodes.
 * It simulates the on-chain logic of the FLASH-IMMUNE framework.
 */
contract FlashImmuneValidator {
    struct Node {
        uint256 reputation; // Reputation score, scaled (e.g., 0-100)
        uint256 historicalContribution; // H(n, t-1)
        uint256 lastUpdateRound;
        bool isQuarantined;
    }

    struct UpdateSubmission {
        uint256 updateNorm; // Statistical fingerprint (L2 norm)
        bytes zkpProof; // The ZKP proof
        bool zkpValid; // Result of ZKP verification
    }

    uint256 public constant REPUTATION_ALPHA = 50; // H weight (50%)
    uint256 public constant REPUTATION_BETA = 30; // C weight (30%)
    uint256 public constant REPUTATION_GAMMA = 20; // V weight (20%)
    uint256 public constant INITIAL_REPUTATION = 100;
    uint256 public constant REPUTATION_THRESHOLD = 30; // Below this, node is quarantined
    uint256 public constant MAX_CONSENSUS_DEVIATION = 50; // 50% deviation allowed

    address public owner;
    uint256 public currentRound;
    bytes32 public globalModelFingerprint;

    IZKPVerifier public immutable ZKP_VERIFIER;

    mapping(address => Node) public nodes;
    mapping(uint256 => address[]) public roundSubmitters;
    mapping(uint256 => mapping(address => UpdateSubmission)) public submissions;

    event NodeRegistered(address indexed node);
    event UpdateSubmitted(
        address indexed node,
        uint256 indexed round,
        uint256 norm,
        bool zkpValid
    );
    event ReputationUpdated(
        address indexed node,
        uint256 newReputation,
        uint256 consensusScore
    );
    event NodeQuarantined(address indexed node, uint256 reputation);
    event NodeRestored(address indexed node, uint256 reputation);
    event RoundFinalized(uint256 indexed round, uint256 medianNorm);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    modifier isRegistered(address node) {
        require(nodes[node].lastUpdateRound > 0, "Node not registered");
        _;
    }

    constructor(address _zkpVerifierAddress) {
        owner = msg.sender;
        ZKP_VERIFIER = IZKPVerifier(_zkpVerifierAddress);
    }

    function registerNode() external {
        require(nodes[msg.sender].lastUpdateRound == 0, "Node already registered");
        nodes[msg.sender] = Node({
            reputation: INITIAL_REPUTATION,
            historicalContribution: 50,
            lastUpdateRound: 1,
            isQuarantined: false
        });
        emit NodeRegistered(msg.sender);
    }

    function startNextRound(bytes32 _newGlobalModelFingerprint)
        external
        onlyOwner
    {
        currentRound++;
        globalModelFingerprint = _newGlobalModelFingerprint;
    }

    /**
     * @notice PHASE 1: Client submits their update, norm, and ZKP.
     * @dev This simulates the client-side submission.
     * @param _updateNorm The statistical norm (fingerprint) of the model delta.
     * @param _zkpProof The ZKP proving valid local training.
     */
    function submitUpdate(uint256 _updateNorm, bytes calldata _zkpProof)
        external
        isRegistered(msg.sender)
    {
        Node storage node = nodes[msg.sender];
        require(!node.isQuarantined, "Node is quarantined");
        require(
            node.lastUpdateRound < currentRound,
            "Already submitted this round"
        );
        require(
            submissions[currentRound][msg.sender].updateNorm == 0,
            "Duplicate submission"
        );

        bytes32 publicInputs = keccak256(
            abi.encodePacked(globalModelFingerprint, _updateNorm)
        );

        bool zkpValid = ZKP_VERIFIER.verify(_zkpProof, publicInputs);

        submissions[currentRound][msg.sender] = UpdateSubmission(
            _updateNorm,
            _zkpProof,
            zkpValid
        );
        roundSubmitters[currentRound].push(msg.sender);
        node.lastUpdateRound = currentRound;

        emit UpdateSubmitted(msg.sender, currentRound, _updateNorm, zkpValid);
    }

    /**
     * @notice PHASE 2: Aggregator finalizes the round.
     * @dev This is called by the owner/aggregator after computing the median norm off-chain.
     * @param _medianNorm The median norm of all valid submissions, computed off-chain.
     */
    function finalizeRound(uint256 _medianNorm) external onlyOwner {
        address[] storage submitters = roundSubmitters[currentRound];

        for (uint i = 0; i < submitters.length; i++) {
            address submitter = submitters[i];
            Node storage node = nodes[submitter];
            UpdateSubmission storage update = submissions[currentRound][submitter];

            uint256 V_nt = update.zkpValid ? 100 : 0;
            uint256 C_nt = _calculateConsensus(update.updateNorm, _medianNorm);
            uint256 H_nt_minus_1 = node.historicalContribution;

            uint256 R_nt = (REPUTATION_ALPHA * H_nt_minus_1 +
                REPUTATION_BETA * C_nt +
                REPUTATION_GAMMA * V_nt) / 100;

            node.reputation = R_nt;
            node.historicalContribution = (H_nt_minus_1 + R_nt) / 2;

            if (R_nt < REPUTATION_THRESHOLD) {
                if (!node.isQuarantined) {
                    node.isQuarantined = true;
                    emit NodeQuarantined(submitter, R_nt);
                }
            } else if (node.isQuarantined) {
                node.isQuarantined = false;
                emit NodeRestored(submitter, R_nt);
            }

            if (V_nt == 100 && C_nt > 50) {
                _sendReward(submitter, R_nt);
            }

            emit ReputationUpdated(submitter, R_nt, C_nt);
        }

        emit RoundFinalized(currentRound, _medianNorm);
    }

    /**
     * @dev Internal function to calculate consensus score based on deviation from median.
     * @param _updateNorm The node's submitted update norm.
     * @param _medianNorm The round's median norm.
     * @return A consensus score (0-100).
     */
    function _calculateConsensus(uint256 _updateNorm, uint256 _medianNorm)
        internal
        pure
        returns (uint256)
    {
        if (_medianNorm == 0) return 50;

        uint256 diff;
        if (_updateNorm > _medianNorm) {
            diff = _updateNorm - _medianNorm;
        } else {
            diff = _medianNorm - _updateNorm;
        }

        uint256 deviation = (diff * 100) / _medianNorm;

        if (deviation > MAX_CONSENSUS_DEVIATION) {
            return 0;
        }

        return 100 - (deviation * 100) / MAX_CONSENSUS_DEVIATION;
    }

    /**
     * @dev Internal function to simulate reward payment.
     * In a real system, this would interact with an ERC20 token contract.
     */
    function _sendReward(address _to, uint256 _reputation) internal {
        // Simulation: In a real system, this would transfer tokens.
    }
}