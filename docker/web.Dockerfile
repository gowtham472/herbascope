# HerbaScope web image. Build context: apps/web (see docker-compose.yml).
# NEXT_PUBLIC_* values are inlined at build time, so the API URL the *browser* uses is a build arg.

FROM node:24-alpine AS deps
WORKDIR /app
RUN corepack enable
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile

FROM node:24-alpine AS build
WORKDIR /app
RUN corepack enable
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL} \
    NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN pnpm build

FROM node:24-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0
RUN addgroup -S herbascope && adduser -S herbascope -G herbascope
# `output: "standalone"` (next.config.ts) emits a self-contained server; static assets are copied alongside.
COPY --from=build --chown=herbascope:herbascope /app/.next/standalone ./
COPY --from=build --chown=herbascope:herbascope /app/.next/static ./.next/static
# The standalone server does not bundle public/, and the brand artwork lives there.
COPY --from=build --chown=herbascope:herbascope /app/public ./public
USER herbascope
EXPOSE 3000
CMD ["node", "server.js"]
