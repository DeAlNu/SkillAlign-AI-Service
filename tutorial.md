Tutorial run backend + AI service
1. git clone https://github.com/randyAhmSya/Chapstone_Project
2. cd Chapstone_Project
3. npm install
4. Buat .env berdasarkan .env.example
  - PROJECT-REF didapat dari project id [https://supabase.com/dashboard/project/tljvgberdcbywhqqphar/settings/general](https://supabase.com/dashboard/project/tljvgberdcbywhqqphar/settings/general)
  - PASSWORD-DATABASE didapat dari password project supabase
  - SUPABASE_SERVICE_ROLE_KEY di dapat dari *service_role* di [https://supabase.com/dashboard/project/tljvgberdcbywhqqphar/settings/api-keys/legacy](https://supabase.com/dashboard/project/tljvgberdcbywhqqphar/settings/api-keys/legacy)
5. npx prisma db push
6. npx prisma generate
7. node prisma/seed.js
8. Buat bucket dengan nama `cv-uploads` di Supabase Storage setting public
9. npm run dev
