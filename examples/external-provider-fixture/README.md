# External provider integration fixture

This is a deliberately separate Gradle build. It is not included in Aljabr's
multi-project build and the provider module has no dependency on Aljabr
implementation modules.

The fixture demonstrates that an independently developed provider can:

1. Compile against the published `aljabr-spi-core` artifact only.
2. Register through `META-INF/services/tech.kayys.aljabr.spi.provider.AljabrProvider`.
3. Be discovered by a host using Java `ServiceLoader`.
4. Receive an immutable `ProviderContext`, expose a descriptor, initialize,
   execute a small operation, and close.

## Run locally

Use JDK 25, matching the current Aljabr SPI toolchain.

From the Aljabr repository root, publish the SPI artifact to the local Maven
repository:

```bash
./gradlew :core:aljabr-spi-core:publishToMavenLocal
```

Then build and run this independent fixture:

```bash
./gradlew -p examples/external-provider-fixture :provider:clean :host:run
```

If testing a different Aljabr version, override the fixture's default
`aljabrVersion`:

```bash
./gradlew -p examples/external-provider-fixture -PaljabrVersion=YOUR_VERSION :host:run
```

Expected output includes:

```text
example/greeting -> Hello, Aljabr!
```

The fixture intentionally uses Maven Local so the provider build resolves the
same published artifact an external repository would consume. It does not use
project substitution, source-set sharing, or implementation project
dependencies.

This is an integration example, not a security sandbox. Java `ServiceLoader`
loads trusted code in-process.
