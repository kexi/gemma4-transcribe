{
  description = "gemma4-transcribe: in-browser speech transcription with Gemma 4 E2B on WebGPU";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      treefmt-nix,
      ...
    }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);

      # One definition drives `nix fmt`, `nix flake check`, `just fmt` and the
      # lefthook hook, so a file can never be formatted one way locally and
      # judged another way in CI.
      treefmtFor =
        pkgs:
        treefmt-nix.lib.evalModule pkgs {
          projectRootFile = "flake.nix";

          programs.nixfmt.enable = true;
          programs.shfmt.enable = true;
          programs.shfmt.indent_size = 4;
          programs.shellcheck.enable = true;

          # Why oxfmt from nixpkgs rather than node_modules: `nix flake check`
          # runs in a sandbox without node_modules, and keeping a second copy
          # in package.json would let the two versions disagree.
          programs.oxfmt.enable = true;

          # .envrc is direnv's stdlib DSL, not a standalone script (no shebang,
          # `use flake` only exists once direnv has sourced its stdlib).
          #
          # Why not treefmt for the justfile: `just --fmt` is still behind
          # --unstable and has no check-only mode treefmt can wrap safely, so
          # it is gated separately by `just fmt-check`.
          settings.global.excludes = [
            ".envrc"
            "justfile"
            "*.lock"
            "pnpm-lock.yaml"
            "LICENSE"
            "dist/*"
          ];
        };
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = [
              # corepack is left out deliberately -- it would fetch a second,
              # unpinned pnpm over the network and defeat the lock.
              pkgs.nodejs_24
              pkgs.pnpm
              pkgs.just
              pkgs.lefthook
              pkgs.gitleaks
              pkgs.pinact
              pkgs.actionlint
              pkgs.shellcheck
              (treefmtFor pkgs).config.build.wrapper
            ];

            shellHook = ''
              # Idempotent: lefthook rewrites .git/hooks on every run.
              if [ -d .git ]; then
                lefthook install >/dev/null 2>&1 || true
              fi
            '';
          };
        }
      );

      formatter = forAllSystems (
        system: (treefmtFor nixpkgs.legacyPackages.${system}).config.build.wrapper
      );

      checks = forAllSystems (system: {
        formatting = (treefmtFor nixpkgs.legacyPackages.${system}).config.build.check self;
      });
    };
}
