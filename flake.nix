{
  description = "eigora — physics simulation library";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems =
        f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs:
        let
          python = pkgs.python313;

          # Runtime deps come from pyproject; the rest is the `dev` extra.
          pythonEnv = python.withPackages (ps: with ps; [
            numpy
            scipy
            pytest
            pytest-cov
            # Only the examples plot; the library itself never imports it.
            matplotlib
          ]);
        in
        {
          default = pkgs.mkShell {
            packages = [ pythonEnv ];

            # src-layout, and no install step: `import eigora` picks up the
            # working tree directly, so edits are live without a reinstall.
            shellHook = ''
              export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
            '';
          };
        });

      packages = forAllSystems (pkgs: {
        default = pkgs.python313Packages.buildPythonPackage {
          pname = "eigora";
          version = "0.1.0";
          src = self;
          pyproject = true;
          build-system = [ pkgs.python313Packages.hatchling ];
          dependencies = with pkgs.python313Packages; [ numpy scipy ];
          # pytest-cov is needed even to switch coverage off: the addopts in
          # pyproject.toml pass --cov unconditionally, so without the plugin
          # pytest rejects the flags rather than ignoring them.
          nativeCheckInputs = with pkgs.python313Packages; [
            pytestCheckHook
            pytest-cov
          ];
          # Coverage is for the dev loop, not for a build.
          pytestFlags = [ "--no-cov" ];
        };
      });
    };
}
