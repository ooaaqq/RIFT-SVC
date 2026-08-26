{
  description = "RIFT-SVC inference development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    {
      self,
      nixpkgs,
      ...
    }:
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
      packages = forAllSystems (
        { pkgs }:
        let
          python = pkgs.python313.withPackages (
            pythonPackages: with pythonPackages; [
              fastapi
              python-multipart
              uvicorn
            ]
          );
          mkEntrypoint =
            name: module:
            pkgs.writeShellApplication {
              inherit name;
              runtimeInputs = [
                pkgs.ffmpeg
                pkgs.kaggle
                python
              ];
              text = ''
                export PYTHONPATH=${self}
                export RIFT_WEB_SOURCE_ROOT=${self}
                exec ${python}/bin/python -m ${module} "$@"
              '';
            };
          package = pkgs.symlinkJoin {
            name = "rift-web-0.1.0";
            paths = [
              (mkEntrypoint "rift-web" "rift_web.serve")
              (mkEntrypoint "rift-dispatcher" "rift_web.dispatcher")
              (mkEntrypoint "rift-cleanup" "rift_web.cleanup")
            ];
          };
        in
        {
          default = package;
          rift-web = package;
        }
      );

      checks = forAllSystems (
        { pkgs }:
        let
          package = self.packages.${pkgs.stdenv.hostPlatform.system}.rift-web;
        in
        {
          rift-web-package = pkgs.runCommand "check-rift-web-package" { } ''
            test -x ${package}/bin/rift-web
            test -x ${package}/bin/rift-dispatcher
            test -x ${package}/bin/rift-cleanup
            touch "$out"
          '';
        }
      );

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
