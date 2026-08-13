{
  description = "RIFT-SVC inference development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      forAllSystems =
        function:
        nixpkgs.lib.genAttrs systems (
          system:
          function {
            pkgs = import nixpkgs { inherit system; };
          }
        );
    in
    {
      devShells = forAllSystems (
        { pkgs }:
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              ffmpeg
              git
              jq
              kaggle
              python311
              ruff
              uv
            ];

            shellHook = ''
              export PATH="${pkgs.python311}/bin:$PATH"
              export UV_PYTHON="${pkgs.python311}/bin/python3.11"
              export PYTHONPATH="$PWD:''${PYTHONPATH:-}"
              export LD_LIBRARY_PATH="${
                pkgs.lib.makeLibraryPath [
                  pkgs.stdenv.cc.cc.lib
                  pkgs.zlib
                ]
              }:''${LD_LIBRARY_PATH:-}"
              export RIFT_SVC_DEV_SHELL=1
              echo "RIFT-SVC dev shell ready: compiler, ffmpeg, kaggle, ruff, uv, Python 3.11"
            '';
          };
        }
      );

      formatter = forAllSystems ({ pkgs }: pkgs.nixfmt-tree);
    };
}
