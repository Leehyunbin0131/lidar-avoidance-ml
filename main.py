"""Entry point for the 2D LiDAR Obstacle Avoidance ML Pipeline."""

from app import LidarPipelineApp


def main() -> None:
    app = LidarPipelineApp()
    app.run()


if __name__ == "__main__":
    main()
