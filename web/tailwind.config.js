/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0a0a0c",
        panel: "#13131a",
        muted: "#6b7280",
        accent: "#7c5cff",
      },
    },
  },
  plugins: [],
};
