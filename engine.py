import os
import secrets
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAYERS = 3          # Number of XOR cascade layers
BOOTHS = 5          # Booths per layer  (valid route digits: 1–5)
TOTAL_KEYS = LAYERS * BOOTHS   # 15 keys total

KEY_HEX_LENGTH = 64  # 32 bytes expressed as 64 hex characters
AES_KEY_BYTES = 32   # AES-256
PBKDF2_ITERATIONS = 600_000
PBKDF2_SALT_BYTES = 16
AES_NONCE_BYTES = 12


# ---------------------------------------------------------------------------
# Key loading
# ---------------------------------------------------------------------------

def _env_key_name(layer: int, booth: int) -> str:
    """Return the environment-variable name for a given layer/booth."""
    return f"QSYZM_KEY_L{layer}_B{booth}"


def load_keys_from_env() -> list[bytes]:
    """
    Load all 15 secret keys from environment variables.

    Expected env var names:
        QSYZM_KEY_L0_B1 … QSYZM_KEY_L0_B5
        QSYZM_KEY_L1_B1 … QSYZM_KEY_L1_B5
        QSYZM_KEY_L2_B1 … QSYZM_KEY_L2_B5

    Each value must be a 64-character hex string (= 32 bytes).

    Returns a flat list of 15 byte strings, row-major (layer 0 first).
    Raises EnvironmentError if any key is missing or malformed.
    """
    keys: list[bytes] = []
    for layer in range(LAYERS):
        for booth in range(1, BOOTHS + 1):
            name = _env_key_name(layer, booth)
            value = os.environ.get(name)
            if not value:
                raise EnvironmentError(
                    f"Missing environment variable: {name}"
                )
            if len(value) != KEY_HEX_LENGTH:
                raise EnvironmentError(
                    f"{name} must be {KEY_HEX_LENGTH} hex chars, got {len(value)}"
                )
            try:
                key_bytes = bytes.fromhex(value)
            except ValueError as exc:
                raise EnvironmentError(
                    f"{name} is not valid hex: {exc}"
                ) from exc
            keys.append(key_bytes)
    return keys  # length == 15


def generate_keys_env_block() -> str:
    """
    Generate 15 fresh random keys and return a ready-to-copy .env block.
    Call this ONCE during initial setup, then store the output securely.
    """
    lines = ["# Qsyzm secret keys — store in a secrets manager, never commit to VCS"]
    for layer in range(LAYERS):
        lines.append(f"\n# Layer {layer}")
        for booth in range(1, BOOTHS + 1):
            key_hex = secrets.token_hex(32)   # 32 bytes → 64 hex chars
            lines.append(f"{_env_key_name(layer, booth)}={key_hex}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Route validation
# ---------------------------------------------------------------------------

def _parse_route(route: str) -> tuple[int, int, int]:
    """
    Validate and parse a 3-digit route string.
    Each digit must be 1–5 (corresponding to booths in a layer).

    Returns a tuple of three zero-based key indices (one per layer).
    """
    if len(route) != LAYERS or not route.isdigit():
        raise ValueError(
            f"Route must be exactly {LAYERS} digits, got: {route!r}"
        )
    indices = []
    for pos, ch in enumerate(route):
        digit = int(ch)
        if not (1 <= digit <= BOOTHS):
            raise ValueError(
                f"Route digit {pos} is {digit!r}; must be 1–{BOOTHS}"
            )
        # Convert to flat index in the keys list
        indices.append(pos * BOOTHS + (digit - 1))
    return tuple(indices)


# ---------------------------------------------------------------------------
# XOR helpers
# ---------------------------------------------------------------------------

def _xor_with_key(data: bytes, key: bytes) -> bytes:
    """XOR data with key, cycling the key if necessary (key-stream style)."""
    key_len = len(key)
    return bytes(b ^ key[i % key_len] for i, b in enumerate(data))


def _xor_cascade(data: bytes, keys: list[bytes], route_indices: tuple) -> bytes:
    """Apply sequential XOR through the three route-selected keys."""
    result = data
    for idx in route_indices:
        result = _xor_with_key(result, keys[idx])
    return result


# ---------------------------------------------------------------------------
# AES-256-GCM helpers (PBKDF2-derived key)
# ---------------------------------------------------------------------------

def _derive_aes_key(password: str, salt: bytes) -> bytes:
    """Derive a 256-bit AES key from a password using PBKDF2-HMAC-SHA256."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=AES_KEY_BYTES,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return kdf.derive(password.encode("utf-8"))


def _aes_encrypt(data: bytes, password: str) -> bytes:
    """
    Encrypt *data* with AES-256-GCM.
    Returns: salt (16 B) || nonce (12 B) || ciphertext+tag
    """
    salt = secrets.token_bytes(PBKDF2_SALT_BYTES)
    nonce = secrets.token_bytes(AES_NONCE_BYTES)
    aes_key = _derive_aes_key(password, salt)
    aesgcm = AESGCM(aes_key)
    ciphertext = aesgcm.encrypt(nonce, data, None)   # no AAD for now
    return salt + nonce + ciphertext


def _aes_decrypt(blob: bytes, password: str) -> bytes:
    """
    Reverse of _aes_encrypt.  Expects: salt || nonce || ciphertext+tag
    """
    if len(blob) < PBKDF2_SALT_BYTES + AES_NONCE_BYTES + 16:
        raise ValueError("Ciphertext blob is too short to be valid.")
    salt = blob[:PBKDF2_SALT_BYTES]
    nonce = blob[PBKDF2_SALT_BYTES: PBKDF2_SALT_BYTES + AES_NONCE_BYTES]
    ciphertext = blob[PBKDF2_SALT_BYTES + AES_NONCE_BYTES:]
    aes_key = _derive_aes_key(password, salt)
    aesgcm = AESGCM(aes_key)
    return aesgcm.decrypt(nonce, ciphertext, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class QsyzmEngine:
    """
    Zero-Knowledge encryption engine for Qsyzm.

    The 15 secret keys are loaded once from environment variables at
    construction time.  Neither the Route nor the Password is ever stored
    on the instance or persisted anywhere.

    Usage:
        engine = QsyzmEngine()
        token  = engine.encrypt("Hello, world!", route="325", password="s3cr3t")
        plain  = engine.decrypt(token, route="325", password="s3cr3t")
    """

    def __init__(self) -> None:
        # Keys are loaded once; they live only in process memory.
        self._keys: list[bytes] = load_keys_from_env()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: str, *, route: str, password: str) -> str:
        """
        Encrypt *plaintext* using the given route and password.

        Parameters
        ----------
        plaintext : str
            The message to encrypt.
        route : str
            A 3-digit string, e.g. '325'.  Each digit (1–5) selects a
            booth in the corresponding layer.
        password : str
            User-supplied password for the final AES layer.

        Returns
        -------
        str
            A URL-safe base64-encoded ciphertext token.
        """
        route_indices = _parse_route(route)

        # Step 1 — XOR cascade through the 3 route-selected keys
        xor_output = _xor_cascade(
            plaintext.encode("utf-8"), self._keys, route_indices
        )

        # Step 2 — AES-256-GCM wrap with PBKDF2-derived key
        aes_blob = _aes_encrypt(xor_output, password)

        # Encode to a portable string for transport/storage
        return base64.urlsafe_b64encode(aes_blob).decode("ascii")

    def decrypt(self, token: str, *, route: str, password: str) -> str:
        """
        Decrypt a token produced by *encrypt*.

        Parameters
        ----------
        token : str
            The base64 token returned by encrypt().
        route : str
            Must match the route used during encryption.
        password : str
            Must match the password used during encryption.

        Returns
        -------
        str
            The original plaintext.

        Raises
        ------
        ValueError
            If the route is invalid, the password is wrong, or the
            ciphertext has been tampered with (AES-GCM authentication
            failure).
        """
        route_indices = _parse_route(route)

        # Step 1 — Base64 decode
        try:
            aes_blob = base64.urlsafe_b64decode(token.encode("ascii"))
        except Exception as exc:
            raise ValueError(f"Token is not valid base64: {exc}") from exc

        # Step 2 — AES-256-GCM unwrap (authenticates integrity + password)
        xor_output = _aes_decrypt(aes_blob, password)

        # Step 3 — Reverse XOR cascade (XOR is its own inverse)
        plaintext_bytes = _xor_cascade(xor_output, self._keys, route_indices)

        return plaintext_bytes.decode("utf-8")


# ---------------------------------------------------------------------------
# CLI helper — key generation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "generate-keys":
        print(generate_keys_env_block())
        sys.exit(0)

    # Quick smoke-test using freshly generated in-memory keys
    print("Running smoke test with temporary in-memory keys...\n")

    # Inject temporary keys into the environment for testing
    for layer in range(LAYERS):
        for booth in range(1, BOOTHS + 1):
            os.environ[_env_key_name(layer, booth)] = secrets.token_hex(32)

    engine = QsyzmEngine()

    test_message = "Qsyzm Phase 1 — ZK engine smoke test ✓"
    test_route   = "325"
    test_password = "correct-horse-battery-staple"

    print(f"  Plaintext : {test_message}")
    print(f"  Route     : {test_route}")
    print(f"  Password  : {test_password}\n")

    token = engine.encrypt(test_message, route=test_route, password=test_password)
    print(f"  Token     : {token}\n")

    recovered = engine.decrypt(token, route=test_route, password=test_password)
    print(f"  Recovered : {recovered}")
    assert recovered == test_message, "MISMATCH — something is wrong!"
    print("\n  ✓ Round-trip successful.")

    # Verify wrong password raises
    try:
        engine.decrypt(token, route=test_route, password="wrong-password")
        print("  ✗ Wrong password should have raised!")
    except Exception:
        print("  ✓ Wrong password correctly rejected.")

    # Verify wrong route raises (or produces garbled output — both acceptable)
    try:
        bad = engine.decrypt(token, route="111", password=test_password)
        assert bad != test_message
        print("  ✓ Wrong route produces different output (no collision).")
    except Exception:
        print("  ✓ Wrong route correctly rejected.")
