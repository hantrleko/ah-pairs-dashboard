import { defineRailway, project, service, volume } from "railway/iac";

// This repository manages only its own resources in the environment. Other
// repositories export their own partial name.
// See https://docs.railway.com/infrastructure-as-code#multi-repo-projects
export const partial = "ah-pairs-dashboard";

export default defineRailway(() => {
  const cache = volume("ah-pairs-cache", {
    region: "sfo",
    sizeMB: 512,
  });
  const ahPairsDashboard = service("ah-pairs-dashboard", {
    healthcheck: "/_stcore/health",
    healthcheckTimeout: 300,
    replicas: { sfo: 1 },
    volumeMounts: {
      "/app/data/cache": cache,
    },
  });

  return project("ah-pairs-dashboard", {
    variables: { managed: false },
    resources: [ahPairsDashboard, cache],
  });
});
