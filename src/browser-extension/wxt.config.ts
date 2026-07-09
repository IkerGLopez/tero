import { defineConfig } from 'wxt';
import vue from "@vitejs/plugin-vue";
import vueI18n from '@intlify/unplugin-vue-i18n/vite';
import tailwindcss from "@tailwindcss/vite";
import Components from 'unplugin-vue-components/vite';
import { PrimeVueResolver } from '@primevue/auto-import-resolver'


export default defineConfig({
    outDir: "dist",
    manifest: () => ({
        name: process.env.EXTENSION_NAME || "Dev Tero Copilot",
        version: process.env.EXTENSION_VERSION,
        key: !process.env.EXTENSION_NAME ? "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAzisSJ8Q4l5mP5Cn1rmVxRCmf3vpXNgvFND7MfNe2WTqkAH4ONopZriBw7cGjaZ82rzDeLDFSfc2DVuwGsR0E4sD+qkwt3ey3LuVd1yeZEid8IbwfeWWYFZXxDmALTcTXcvLmDdMeNPhkPjjd4+e5QtaO9ufEzps/11NTOhaCa0nJ7vR+U3OMV/iWMkqShNy3dt9QUWOA78B2U4tbgEtAz+cns0I4/TW57E8lqJJsKVh3S3tcqGmcPjJLOATVHLJXzFe0ABVGYnXjyvT80msaWS5SSV3/7mSD90GDOapJMnvs//ksnRh3EG4bKKtQ9sDmXJVkvS4NYVvt+ozjNuIlyQIDAQAB" : undefined,
        host_permissions: ["http://*/*", "https://*/*"],
        web_accessible_resources: [{
            matches: ['http://*/*', 'https://*/*'],
            resources: ['iframe.html']
        }],
        permissions: [
            "activeTab",
            "storage",
            "contextMenus",
            "webRequest",
            "declarativeNetRequest",
            "clipboardWrite",
            "identity",
            "tabs"
        ],
        action: {}
    }),
    webExt: {
        startUrls: [process.env.START_URL || "https://github.com/abstracta/tero"]
    },
    vite: () => ({
        esbuild: {
            supported: {
                'top-level-await': true
            }
        },
        plugins: [
            vue(),
            tailwindcss(),
            Components({
                dts: true,
                dirs: ['components', '../common/src'],
                resolvers: [PrimeVueResolver()]
            }),
            vueI18n({}),
        ],
    }),
});
