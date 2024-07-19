let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
      pkgs.python3
      pkgs.poetry

      pkgs.libstdcxx5
      pkgs.zlib
      pkgs.libGL
      pkgs.glib
  ];

  LD_LIBRARY_PATH = "${pkgs.libstdcxx5}/lib:${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.libGL}/lib";
}
