import { defineConfig } from "vite";

// Office loads the pane as a plain web page, so the bundle is a self-contained ES module
// served over HTTPS. No framework: the pane is five states and a form, and Alpine.js is
// added from the page rather than the build.
export default defineConfig({
  build: {
    target: "es2022",
    outDir: "dist",
    rollupOptions: { input: { taskpane: "src/taskpane.html" } },
  },
  test: {
    globals: true,
    environment: "node",
    include: ["test/**/*.test.ts"],
    coverage: { include: ["src/**/*.ts"], exclude: ["src/taskpane.ts"] },
  },
});
