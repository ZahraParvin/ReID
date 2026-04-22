import sys

if __name__ == "__main__":
    # Default to Camera-2; can still be overridden: python edge_client_2.py --camera-id Camera-X
    if "--camera-id" not in sys.argv:
        sys.argv += ["--camera-id", "Camera-2"]
    from edge_client import main
    main()
