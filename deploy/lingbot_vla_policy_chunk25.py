"""Chunk-25 WebSocket server for LingBot-VLA.

This is a separate entry point for chunk-25 checkpoints.  The original
``lingbot_vla_policy`` entry point remains unchanged and keeps its historical
chunk-50 behavior.
"""

from __future__ import annotations

import argparse

from . import lingbot_vla_policy as base


class LingbotVlaChunk25Server(base.LingbotVLAServer):
    """Inference server with an explicit 25-frame model/normalizer contract."""

    CHUNK_SIZE = 25

    def __init__(self, *args, **kwargs):
        use_length = kwargs.get("use_length", self.CHUNK_SIZE)
        if use_length != self.CHUNK_SIZE:
            raise ValueError(
                f"chunk25 server requires --use_length {self.CHUNK_SIZE}; "
                f"got {use_length}"
            )
        super().__init__(*args, **kwargs)

    def load_vla(self, path_to_pi_model):
        # The base loader only fills config fields that are absent.  Qwen's
        # saved config can therefore leave chunk_size=50 even for a chunk25
        # checkpoint.  Override it after the base loader has built both
        # namespaces, before reset() creates FeatureTransform.
        vla = super().load_vla(path_to_pi_model)
        self.config.chunk_size = self.CHUNK_SIZE
        self.data_config.chunk_size = self.CHUNK_SIZE
        print(f"Using explicit chunk_size={self.CHUNK_SIZE}")
        return vla


def main() -> None:
    parser = argparse.ArgumentParser(description="Start chunk-25 WebSocket policy server")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--use_length", type=int, default=25,
        help="Execution length; chunk25 server requires exactly 25",
    )
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--norm_path", type=str, default=None)
    parser.add_argument("--num_denoising_step", type=int, default=10)
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--rtc-mode", choices=("off", "true"), default="off")
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=5.0)
    args = parser.parse_args()

    if args.use_length != LingbotVlaChunk25Server.CHUNK_SIZE:
        parser.error("--use_length must be 25 for the chunk25 server")

    model = LingbotVlaChunk25Server(
        args.model_path,
        use_length=args.use_length,
        robot_norm_path=args.norm_path,
        num_denoising_step=args.num_denoising_step,
        use_compile=args.use_compile,
        rtc_mode=args.rtc_mode,
        rtc_max_guidance_weight=args.rtc_max_guidance_weight,
    )
    model_server = base.WebsocketPolicyServer(model, port=args.port)
    model_server.serve_forever()


if __name__ == "__main__":
    main()
