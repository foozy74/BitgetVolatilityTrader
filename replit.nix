{pkgs}: {
  deps = [
    pkgs.openssh
    pkgs.glibcLocales
    pkgs.postgresql
    pkgs.openssl
  ];
}
