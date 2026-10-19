import argparse

def run_experiment_cli(main_func):
    parser = argparse.ArgumentParser()

    parser.add_argument( "--viewer", dest="use_viewer", action="store_true", help="Run simulation with viewer", )
    parser.add_argument( "--no-viewer", dest="use_viewer", action="store_false", help="Run simulation without viewer", )
    parser.add_argument( "--plot", dest="generate_plot", action="store_true", help="Generate plots after simulation", )
    parser.add_argument( "--no-plot", dest="generate_plot", action="store_false", help="Skip plot generation", )
    parser.add_argument( "--gif", dest="generate_gif", action="store_true", help="Generate GIF after simulation", )
    parser.add_argument( "--no-gif", dest="generate_gif", action="store_false", help="Skip GIF generation", )

    parser.set_defaults( use_viewer=True, generate_plot=True, generate_gif=True, )

    args = parser.parse_args()

    print(f"Use viewer: {args.use_viewer}")
    print(f"Generate plot: {args.generate_plot}")
    print(f"Generate GIF: {args.generate_gif}")

    main_func(
        use_viewer=args.use_viewer,
        generate_plot=args.generate_plot,
        generate_gif=args.generate_gif,
    )